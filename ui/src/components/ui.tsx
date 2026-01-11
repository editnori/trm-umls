import type { KeyboardEventHandler, ReactNode } from "react";
import { forwardRef } from "react";
import { cx } from "./cx";

export function Card(props: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cx("ui-card", props.className)}>
      <div className="ui-card-header">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold tracking-tight text-slate-100">
            {props.title}
          </div>
          {props.subtitle ? <div className="mt-0.5 text-xs text-slate-400">{props.subtitle}</div> : null}
        </div>
        {props.right ? <div className="shrink-0">{props.right}</div> : null}
      </div>
      <div className="ui-card-body">{props.children}</div>
    </section>
  );
}

export function Button(props: {
  variant?: "primary" | "secondary" | "ghost";
  disabled?: boolean;
  onClick?: () => void;
  children: ReactNode;
  className?: string;
  type?: "button" | "submit";
}) {
  const v = props.variant ?? "secondary";
  return (
    <button
      type={props.type ?? "button"}
      className={cx(
        "ui-button",
        v === "primary" ? "ui-button-primary" : v === "ghost" ? "ui-button-ghost" : "ui-button-secondary",
        props.disabled ? "pointer-events-none opacity-50" : "",
        props.className,
      )}
      onClick={props.onClick}
      disabled={props.disabled}
    >
      {props.children}
    </button>
  );
}

export function Pill(props: { children: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={props.title}
      className={cx(
        "inline-flex items-center rounded-md border px-2 py-1 text-[11px] font-semibold",
        "border-white/10 bg-slate-950/40 text-slate-200",
        props.className,
      )}
    >
      {props.children}
    </span>
  );
}

export function Switch(props: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  description?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={props.checked}
      onClick={() => props.onChange(!props.checked)}
      className="flex w-full items-center justify-between gap-3 rounded-lg border border-white/10 bg-slate-950/35 px-3 py-2 text-left hover:bg-slate-950/55"
    >
      <div className="min-w-0">
        <div className="truncate text-sm font-semibold text-slate-200">{props.label}</div>
        {props.description ? <div className="mt-0.5 text-xs text-slate-400">{props.description}</div> : null}
      </div>
      <span
        className={cx(
          "relative inline-flex h-6 w-10 shrink-0 items-center rounded-full border transition-colors",
          props.checked ? "border-sky-400/35 bg-sky-500/25" : "border-white/10 bg-slate-950/60",
        )}
      >
        <span
          className={cx(
            "h-5 w-5 rounded-full bg-slate-200 transition-transform",
            props.checked ? "translate-x-[18px]" : "translate-x-[2px]",
          )}
        />
      </span>
    </button>
  );
}

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
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-slate-950/35 px-3 py-2">
      <div className="min-w-0">
        <div className="truncate text-sm font-semibold text-slate-200">{props.label}</div>
        <div className="mt-0.5 text-xs text-slate-400">
          min {props.min} · max {props.max}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <button
          type="button"
          className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-white/10 bg-slate-950/60 text-slate-200 hover:bg-slate-950/80"
          onClick={() => props.onChange(clamp(props.value - props.step, props.min, props.max))}
          aria-label={`decrease ${props.label}`}
        >
          −
        </button>
        <div className="min-w-[72px] rounded-md border border-white/10 bg-slate-950/55 px-2 py-1 text-center font-mono text-xs tabular-nums text-slate-200">
          {shown}
        </div>
        <button
          type="button"
          className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-white/10 bg-slate-950/60 text-slate-200 hover:bg-slate-950/80"
          onClick={() => props.onChange(clamp(props.value + props.step, props.min, props.max))}
          aria-label={`increase ${props.label}`}
        >
          +
        </button>
      </div>
    </div>
  );
}

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
      className={cx("ui-input font-mono leading-6 text-slate-100", props.className)}
      value={props.value}
      onChange={(e) => props.onChange(e.target.value)}
      placeholder={props.placeholder}
      onKeyDown={props.onKeyDown}
      spellCheck={false}
      rows={props.rows ?? 12}
    />
  );
});

function clamp(n: number, min: number, max: number): number {
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, n));
}
