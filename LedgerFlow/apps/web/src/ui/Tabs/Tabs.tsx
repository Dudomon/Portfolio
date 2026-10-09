import {
  createContext,
  useContext,
  useId,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';

import { cx } from '@/lib/cx';

import styles from './Tabs.module.css';

interface TabsContextValue {
  baseId: string;
  value: string;
  select: (value: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabsContext(component: string): TabsContextValue {
  const context = useContext(TabsContext);
  if (!context) throw new Error(`<${component}> must be rendered inside <Tabs>`);
  return context;
}

const toIdPart = (value: string) => value.replace(/[^\w-]/g, '_');

export interface TabsProps {
  /** Controlled selected tab. */
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  children: ReactNode;
  className?: string;
}

/**
 * WAI-ARIA tabs with automatic activation: arrow keys move focus and select,
 * Home/End jump to the ends, and only the selected tab is in the tab order.
 */
export function Tabs({ value, defaultValue = '', onValueChange, children, className }: TabsProps) {
  const baseId = useId();
  const [uncontrolled, setUncontrolled] = useState(defaultValue);
  const selected = value ?? uncontrolled;

  const select = (next: string) => {
    if (value === undefined) setUncontrolled(next);
    onValueChange?.(next);
  };

  return (
    <TabsContext.Provider value={{ baseId, value: selected, select }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
}

const NAVIGATION_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'Home', 'End']);

function nextTabIndex(key: string, current: number, count: number): number {
  switch (key) {
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    case 'ArrowRight':
      return (current + 1) % count;
    default:
      return (current - 1 + count) % count;
  }
}

export function TabList({
  'aria-label': ariaLabel,
  children,
  className,
}: {
  'aria-label': string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      aria-orientation="horizontal"
      className={cx(styles.list, className)}
    >
      {children}
    </div>
  );
}

function onTabKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
  if (!NAVIGATION_KEYS.has(event.key)) return;
  const list = event.currentTarget.closest('[role="tablist"]');
  if (!list) return;
  const tabs = Array.from(list.querySelectorAll<HTMLButtonElement>('[role="tab"]:not(:disabled)'));
  const current = tabs.indexOf(event.currentTarget);
  if (current === -1) return;
  event.preventDefault();
  const next = tabs[nextTabIndex(event.key, current, tabs.length)];
  next?.focus();
  next?.click();
}

export function Tab({
  value,
  disabled,
  children,
}: {
  value: string;
  disabled?: boolean;
  children: ReactNode;
}) {
  const { baseId, value: selectedValue, select } = useTabsContext('Tab');
  const selected = value === selectedValue;
  const idPart = toIdPart(value);
  return (
    <button
      type="button"
      role="tab"
      id={`${baseId}-tab-${idPart}`}
      aria-selected={selected}
      aria-controls={`${baseId}-panel-${idPart}`}
      tabIndex={selected ? 0 : -1}
      disabled={disabled}
      className={styles.tab}
      onClick={() => select(value)}
      onKeyDown={onTabKeyDown}
    >
      {children}
    </button>
  );
}

export function TabPanel({
  value,
  keepMounted = false,
  children,
  className,
}: {
  value: string;
  /** Keep inactive content mounted (hidden) so local state such as form input survives. */
  keepMounted?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const { baseId, value: selectedValue } = useTabsContext('TabPanel');
  const selected = value === selectedValue;
  if (!selected && !keepMounted) return null;
  const idPart = toIdPart(value);
  return (
    <div
      role="tabpanel"
      id={`${baseId}-panel-${idPart}`}
      aria-labelledby={`${baseId}-tab-${idPart}`}
      hidden={!selected}
      tabIndex={0}
      className={cx(styles.panel, className)}
    >
      {children}
    </div>
  );
}
