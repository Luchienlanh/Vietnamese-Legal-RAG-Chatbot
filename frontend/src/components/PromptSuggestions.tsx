import { Scale, X } from "lucide-react";

interface PromptSuggestionsProps {
  prompts: string[];
  onSelect: (prompt: string) => void;
  compact?: boolean;
  onDismiss?: () => void;
}

export function PromptSuggestions({
  prompts,
  onSelect,
  compact = false,
  onDismiss,
}: PromptSuggestionsProps) {
  return (
    <div className={compact ? "prompt-suggestions compact" : "prompt-suggestions"}>
      <div className="prompt-suggestions__header">
        <div className="prompt-suggestions__title">
          <Scale aria-hidden="true" size={16} />
          <span>Gợi ý tra cứu</span>
        </div>
        {onDismiss ? (
          <button
            aria-label="Đóng gợi ý tra cứu"
            className="icon-button prompt-suggestions__dismiss"
            onClick={onDismiss}
            type="button"
          >
            <X aria-hidden="true" size={15} />
          </button>
        ) : null}
      </div>
      <div className="prompt-suggestions__grid">
        {prompts.map((prompt) => (
          <button
            className="prompt-button"
            key={prompt}
            onClick={() => onSelect(prompt)}
            type="button"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
