import { useEffect, useRef } from 'react';

export type HotkeyMap = Partial<Record<string, (event: KeyboardEvent) => void>>;

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  return target.closest('input, textarea, select, [role="textbox"]') !== null;
}

/**
 * Single-key shortcuts that stay out of the way: ignored while typing in a
 * form control, with a modifier held, or while a modal dialog is open.
 */
export function useHotkeys(hotkeys: HotkeyMap, enabled = true): void {
  const hotkeysRef = useRef(hotkeys);
  useEffect(() => {
    hotkeysRef.current = hotkeys;
  });

  useEffect(() => {
    if (!enabled) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented || event.repeat) return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (isEditableTarget(event.target)) return;
      if (document.querySelector('[aria-modal="true"]')) return;
      const handler = hotkeysRef.current[event.key.toLowerCase()];
      if (handler) {
        event.preventDefault();
        handler(event);
      }
    }
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [enabled]);
}
