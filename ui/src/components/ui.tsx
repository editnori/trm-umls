import type { KeyboardEventHandler, ReactNode } from "react";
import { forwardRef } from "react";
import { cx } from "./cx";

/**
 * UI Components - OpenCode style
 * Monochrome, squarish, minimal
 */

// Button - Monochrome
export function Button(props: {
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  onClick?: () => void;
  children: ReactNode;
  className?: string;
  type?: "button" | "submit";
}) {
  const v = props.variant ?? "secondary";
  
  const styles = {
    primary: "bg-[var(--text-primary)] text-[var(--bg-page)] hover:opacity-90",
    secondary: "bg-[var(--bg-elevated)] text-[var(--text-body)] border border-[var(--border)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]",
    ghost: "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]",
  };

  return (
    <button
      type={props.type ?? "button"}
      className={cx(
        "inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded transition-all",
        styles[v],
        props.disabled && "opacity-30 pointer-events-none",
        props.className,
      )}
      onClick={props.onClick}
      disabled={props.disabled}
    >
      {props.children}
    </button>
  );
}

// Stepper - Compact, squarish
export function Stepper(props: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step: number;
  min: number;
  max: number;
  format?: (v: number) => string;
}) {
  const shown = props.format ? props.format(props.value) : String(props.value);
  const clamp = (n: number) => Math.max(props.min, Math.min(props.max, n));
  
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-[11px] text-[var(--text-body)]">{props.label}</span>
      <div className="flex items-center bg-[var(--bg-elevated)] rounded border border-[var(--border)]">
        <button
          type="button"
          className="w-6 h-6 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors text-xs"
          onClick={() => props.onChange(clamp(props.value - props.step))}
        >
          −
        </button>
        <span className="w-9 text-center font-mono text-[11px] font-medium tabular-nums text-[var(--text-primary)]">
          {shown}
        </span>
        <button
          type="button"
          className="w-6 h-6 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors text-xs"
          onClick={() => props.onChange(clamp(props.value + props.step))}
        >
          +
        </button>
      </div>
    </div>
  );
}

// Switch - Monochrome with label
export function Switch(props: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
}) {
  return (
    <label className="flex items-center justify-between py-1 cursor-pointer group">
      <span className="text-[11px] text-[var(--text-body)] group-hover:text-[var(--text-primary)] transition-colors">
        {props.label}
      </span>
      <div className="flex items-center gap-1.5">
        <span className={cx(
          "text-[9px] font-medium uppercase tracking-wide",
          props.checked ? "text-[var(--text-primary)]" : "text-[var(--text-faint)]"
        )}>
          {props.checked ? "on" : "off"}
        </span>
        <div
          className={cx("toggle-track", props.checked ? "on" : "off")}
          onClick={() => props.onChange(!props.checked)}
        >
          <div className="toggle-thumb" />
        </div>
      </div>
    </label>
  );
}

// Textarea - Minimal border
export const Textarea = forwardRef<
  HTMLTextAreaElement,
  {
    value: string;
    onChange: (v: string) => void;
    placeholder?: string;
    className?: string;
    rows?: number;
    onKeyDown?: KeyboardEventHandler<HTMLTextAreaElement>;
  }
>(function TextareaImpl(props, ref) {
  return (
    <textarea
      ref={ref}
      className={cx(
        "w-full px-3 py-2 text-[12px] font-mono leading-relaxed",
        "bg-[var(--bg-elevated)] text-[var(--text-primary)] placeholder:text-[var(--text-faint)]",
        "border border-[var(--border)] rounded",
        "focus:outline-none focus:border-[var(--border-strong)]",
        "resize-none transition-colors",
        props.className,
      )}
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
      placeholder={props.placeholder}
      onKeyDown={props.onKeyDown}
      spellCheck={false}
      rows={props.rows ?? 6}
    />
  );
});
