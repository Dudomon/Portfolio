import ui from '@/components/ui.module.css';

export default function Loading() {
  return (
    <p className={ui.muted} role="status">
      Loading…
    </p>
  );
}
