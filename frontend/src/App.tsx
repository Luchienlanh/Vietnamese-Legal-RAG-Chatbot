import { useMemo, useState } from "react";
import { ChatLayout } from "./components/ChatLayout";
import { ChatThread } from "./components/ChatThread";
import { Sidebar } from "./components/Sidebar";
import { SourcePanel } from "./components/SourcePanel";
import { useTheme } from "./hooks/useTheme";
import {
  promptSuggestions,
  searchLegalAnswer,
  seedConversations,
} from "./services/legalResearchService";
import type {
  ChatMessage,
  Citation,
  Conversation,
  ConversationCategory,
} from "./types/legal";

const createId = (prefix: string) =>
  `${prefix}-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 8)}`;

const normalize = (value: string) =>
  value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();

const wait = (ms: number) =>
  new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const inferCategory = (
  question: string
): Exclude<ConversationCategory, "Tất cả"> => {
  const normalizedQuestion = normalize(question);

  if (normalizedQuestion.includes("lao dong")) {
    return "Lao động";
  }

  if (normalizedQuestion.includes("dat dai")) {
    return "Đất đai";
  }

  if (normalizedQuestion.includes("hanh chinh") || normalizedQuestion.includes("xu phat")) {
    return "Hành chính";
  }

  if (normalizedQuestion.includes("hinh su")) {
    return "Hình sự";
  }

  if (normalizedQuestion.includes("an le")) {
    return "Án lệ";
  }

  return "Dân sự";
};

const createConversation = (): Conversation => ({
  id: createId("conv"),
  title: "Tra cứu mới",
  category: "Dân sự",
  updatedAt: new Date().toISOString(),
  lastSummary: "Chưa có nội dung.",
  messages: [],
});

export default function App() {
  const [conversations, setConversations] =
    useState<Conversation[]>(seedConversations);
  const [activeConversationId, setActiveConversationId] = useState(
    seedConversations[0]?.id ?? ""
  );
  const [historySearch, setHistorySearch] = useState("");
  const [selectedCategory, setSelectedCategory] =
    useState<ConversationCategory>("Tất cả");
  const [selectedCitation, setSelectedCitation] = useState<Citation | undefined>(
    seedConversations[0]?.messages[1]?.citations?.[0]
  );
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSourceOpen, setIsSourceOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { preference, setPreference } = useTheme();

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ??
    conversations[0] ??
    createConversation();

  const filteredConversations = useMemo(() => {
    const normalizedSearch = normalize(historySearch);

    return conversations.filter((conversation) => {
      const matchesCategory =
        selectedCategory === "Tất cả" ||
        conversation.category === selectedCategory;
      const matchesSearch =
        !normalizedSearch ||
        normalize(
          `${conversation.title} ${conversation.category} ${conversation.lastSummary}`
        ).includes(normalizedSearch);

      return matchesCategory && matchesSearch;
    });
  }, [conversations, historySearch, selectedCategory]);

  const updateConversation = (
    conversationId: string,
    updater: (conversation: Conversation) => Conversation
  ) => {
    setConversations((currentConversations) =>
      currentConversations.map((conversation) =>
        conversation.id === conversationId ? updater(conversation) : conversation
      )
    );
  };

  const handleNewChat = () => {
    const nextConversation = createConversation();
    setConversations((currentConversations) => [
      nextConversation,
      ...currentConversations,
    ]);
    setActiveConversationId(nextConversation.id);
    setSelectedCitation(undefined);
    setIsSidebarOpen(false);
  };

  const streamAnswer = async (
    conversationId: string,
    messageId: string,
    fullAnswer: string
  ) => {
    const chunkSize = 12;

    for (let index = chunkSize; index <= fullAnswer.length; index += chunkSize) {
      await wait(24);
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.id === messageId
            ? { ...message, content: fullAnswer.slice(0, index) }
            : message
        ),
      }));
    }

    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      messages: conversation.messages.map((message) =>
        message.id === messageId
          ? { ...message, content: fullAnswer, status: "complete" }
          : message
      ),
    }));
  };

  const handleSend = async (question: string) => {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || isSubmitting) {
      return;
    }

    const conversationId = activeConversation.id;
    const currentMessageCount = activeConversation.messages.length;
    const category = inferCategory(trimmedQuestion);
    const createdAt = new Date().toISOString();
    const userMessage: ChatMessage = {
      id: createId("msg-user"),
      role: "user",
      content: trimmedQuestion,
      status: "complete",
      createdAt,
    };
    const assistantMessage: ChatMessage = {
      id: createId("msg-assistant"),
      role: "assistant",
      content: "",
      status: "loading",
      createdAt,
    };

    setIsSubmitting(true);
    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      title:
        currentMessageCount === 0
          ? trimmedQuestion.slice(0, 52)
          : conversation.title,
      category: currentMessageCount === 0 ? category : conversation.category,
      lastSummary: trimmedQuestion,
      updatedAt: createdAt,
      messages: [...conversation.messages, userMessage, assistantMessage],
    }));

    try {
      const answer = await searchLegalAnswer({
        question: trimmedQuestion,
        topic: category,
        conversationId,
      });

      if (answer.citations[0]) {
        setSelectedCitation(answer.citations[0]);
      }

      if (answer.citations.length === 0) {
        updateConversation(conversationId, (conversation) => ({
          ...conversation,
          messages: conversation.messages.map((message) =>
            message.id === assistantMessage.id
              ? {
                  ...message,
                  content: answer.answer,
                  status: "no-result",
                  confidence: answer.confidence,
                  followUps: answer.followUps,
                }
              : message
          ),
        }));
        setIsSubmitting(false);
        return;
      }

      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.id === assistantMessage.id
            ? {
                ...message,
                content: "",
                status: "streaming",
                citations: answer.citations,
                confidence: answer.confidence,
                followUps: answer.followUps,
              }
            : message
        ),
      }));

      await streamAnswer(conversationId, assistantMessage.id, answer.answer);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Có lỗi không xác định khi tra cứu.";
      updateConversation(conversationId, (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((chatMessage) =>
          chatMessage.id === assistantMessage.id
            ? {
                ...chatMessage,
                content: `${message} Vui lòng thử lại hoặc thu hẹp câu hỏi.`,
                status: "error",
              }
            : chatMessage
        ),
      }));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleConversationSelect = (conversationId: string) => {
    setActiveConversationId(conversationId);
    setIsSidebarOpen(false);
    const conversation = conversations.find((item) => item.id === conversationId);
    setSelectedCitation(conversation?.messages.find((message) => message.citations?.[0])?.citations?.[0]);
  };

  const handleCitationSelect = (citation: Citation) => {
    setSelectedCitation(citation);
    setIsSourceOpen(true);
  };

  return (
    <ChatLayout
      hasSelectedSource={Boolean(selectedCitation)}
      isSidebarOpen={isSidebarOpen}
      isSourceOpen={isSourceOpen}
      onCloseSidebar={() => setIsSidebarOpen(false)}
      onCloseSource={() => setIsSourceOpen(false)}
      onOpenSidebar={() => setIsSidebarOpen(true)}
      onOpenSource={() => setIsSourceOpen(true)}
      sidebar={
        <Sidebar
          activeConversationId={activeConversation.id}
          conversations={filteredConversations}
          onCategoryChange={setSelectedCategory}
          onConversationSelect={handleConversationSelect}
          onNewChat={handleNewChat}
          onSearchChange={setHistorySearch}
          onThemeChange={setPreference}
          searchValue={historySearch}
          selectedCategory={selectedCategory}
          themePreference={preference}
        />
      }
      sourcePanel={
        <SourcePanel
          citation={selectedCitation}
          onClose={() => setIsSourceOpen(false)}
        />
      }
      thread={
        <ChatThread
          conversation={activeConversation}
          isSubmitting={isSubmitting}
          onCitationSelect={handleCitationSelect}
          onSend={handleSend}
          prompts={promptSuggestions}
          selectedCitationId={selectedCitation?.id}
        />
      }
    />
  );
}
