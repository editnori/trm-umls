import type { Assertion, SemanticGroup } from "./types";

/**
 * Entity highlighting - CSS class based for clean styling
 */

// Get entity mark class for highlighted spans
export function entityMarkClass(g: SemanticGroup, isSelected: boolean): string {
  const base = `entity-mark entity-${g}`;
  return isSelected ? `${base} selected` : base;
}

// Get filter pill class
export function filterPillClass(active: boolean): string {
  return active ? "filter-pill" : "filter-pill inactive";
}

// Assertion filter pill
export function assertionPillClass(a: Assertion, active: boolean): string {
  const base = `filter-pill assertion-${a}`;
  return active ? base : `${base} inactive`;
}

// Group filter pill
export function groupPillClass(g: SemanticGroup, active: boolean): string {
  const base = `filter-pill entity-${g}`;
  return active ? base : `${base} inactive`;
}

// Table badge - smaller
export function entityTableBadge(g: SemanticGroup): string {
  return `inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold entity-${g}`;
}

export function assertionTableBadge(a: Assertion): string {
  return `inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold assertion-${a}`;
}
