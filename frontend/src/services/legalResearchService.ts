import type {
  ChatRequest,
  Citation,
  Conversation,
  LegalAnswer,
  SourceType,
} from "../types/legal";

export const promptSuggestions = [
  "Tóm tắt quy định về hợp đồng lao động",
  "Tìm án lệ liên quan đến tranh chấp đất đai",
  "So sánh quy định trong pháp điển và văn bản gốc",
  "Quy trình đăng ký giấy kết hôn",
  "Điều kiện kết hôn theo pháp luật Việt Nam",
];

export const seedConversations: Conversation[] = [
  {
    id: "conv-tra-cuu-moi",
    title: "Tra cứu pháp luật",
    category: "Dân sự",
    updatedAt: new Date().toISOString(),
    lastSummary: "Hỏi về văn bản pháp luật, pháp điển hoặc án lệ.",
    messages: [],
  },
];

interface BackendChatResponse {
  session_id: string;
  answer?: string;
  sources?: BackendSource[];
}

interface BackendSource {
  source_type?: string;
  title?: string;
  source_url?: string;
  retrieval_mode?: string;
  retrieval_modes?: string[];
  context_mode?: string;
  content?: string;
  excerpt?: string;
  metadata?: Record<string, unknown>;
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";
const chatEndpoint = apiBaseUrl ? `${apiBaseUrl}/chat` : "/api/chat";

const asText = (value: unknown): string | undefined => {
  if (value === null || value === undefined) {
    return undefined;
  }

  const text = String(value).trim();
  return text.length ? text : undefined;
};

const sourceTypeFromBackend = (value: unknown): SourceType => {
  const normalized = asText(value)?.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase() ?? "";

  if (normalized.includes("anle") || normalized.includes("an_le") || normalized.includes("an le")) {
    return "an-le";
  }

  if (normalized.includes("phapdien") || normalized.includes("phap_dien") || normalized.includes("phap dien")) {
    return "phap-dien";
  }

  return "van-ban";
};

const articleLabel = (sourceType: SourceType, metadata: Record<string, unknown>) => {
  const originalArticleNumber = asText(metadata.original_article_number);
  const articleTitle = asText(metadata.article_title);
  const subjectTitle = asText(metadata.subject_title);

  if (originalArticleNumber) {
    return originalArticleNumber.toLowerCase().includes("điều")
      ? originalArticleNumber
      : `Điều ${originalArticleNumber}`;
  }

  if (articleTitle) {
    return articleTitle;
  }

  if (sourceType === "phap-dien") {
    return subjectTitle;
  }

  return undefined;
};

const compact = (value: string, maxLength: number) => {
  const normalized = value.replace(/\s+/g, " ").trim();

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 3)}...`;
};

const unique = (values: Array<string | undefined>) =>
  Array.from(new Set(values.filter((value): value is string => Boolean(value))));

const mapBackendSource = (source: BackendSource, index: number): Citation => {
  const metadata = source.metadata ?? {};
  const sourceType = sourceTypeFromBackend(source.source_type);
  const title =
    asText(source.title) ??
    asText(metadata.article_title) ??
    asText(metadata.title) ??
    asText(metadata.doc_name) ??
    asText(metadata.subject_title) ??
    "Nguồn pháp lý";
  const excerpt =
    asText(source.excerpt) ??
    asText(source.content) ??
    "Backend chưa trả về đoạn trích chi tiết cho nguồn này.";
  const retrievalMode = asText(source.retrieval_mode);
  const retrievalModes = source.retrieval_modes?.map(asText).filter((value): value is string => Boolean(value));
  const modeText = retrievalModes?.length ? retrievalModes.join(", ") : retrievalMode;
  const topicTitle = asText(metadata.topic_title);
  const subjectTitle = asText(metadata.subject_title);
  const docCode = asText(metadata.doc_code);

  return {
    id: `${sourceType}-${index}-${title}`,
    sourceType,
    title,
    article: articleLabel(sourceType, metadata),
    effectiveDate: asText(metadata.effective_date) ?? asText(metadata.issue_date),
    status:
      sourceType === "an-le"
        ? "Nguồn án lệ/bản án"
        : sourceType === "phap-dien"
          ? "Theo dữ liệu pháp điển"
          : "Nguồn văn bản",
    agency: asText(metadata.agency) ?? asText(metadata.issuer) ?? asText(metadata.subject),
    detailUrl: asText(source.source_url) ?? asText(metadata.detail_url) ?? asText(metadata.pdf_url),
    summary: compact(
      unique([
        topicTitle,
        subjectTitle,
        docCode,
        modeText ? `Truy xuất: ${modeText}` : undefined,
      ]).join(". ") || excerpt,
      220
    ),
    excerpt: compact(excerpt, 900),
    related: unique([topicTitle, subjectTitle, docCode]).slice(0, 4),
  };
};

export async function searchLegalAnswer(
  request: ChatRequest
): Promise<LegalAnswer> {
  const response = await fetch(chatEndpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message: request.question,
      session_id: request.conversationId,
      k: 8,
      validation_mode: "none",
      include_debug: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`Backend trả về lỗi HTTP ${response.status}.`);
  }

  const data = (await response.json()) as BackendChatResponse;
  const citations = (data.sources ?? []).map(mapBackendSource);

  return {
    answer:
      data.answer?.trim() ||
      "Backend không trả về nội dung trả lời. Hãy thử diễn đạt lại câu hỏi hoặc kiểm tra dữ liệu chỉ mục.",
    citations,
    confidence: citations.length ? "cao" : "thấp",
    followUps: [
      "Liệt kê căn cứ pháp lý chính",
      "Tóm tắt hồ sơ và trình tự thực hiện",
    ],
    noResultReason: citations.length ? undefined : "Backend không trả về nguồn tham chiếu.",
  };
}
