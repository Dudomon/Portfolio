import { describeError, useAudit, type AuditEvent } from '@/api';
import { formatDateTime } from '@/lib/format';
import { Button, Callout, SkeletonText } from '@/ui';

import styles from './AuditTimeline.module.css';

const ACTION_LABEL: Record<string, string> = {
  'document.uploaded': 'Uploaded',
  'document.queued': 'Queued for extraction',
  'document.extracted': 'Fields extracted',
  'document.needs_review': 'Sent to review',
  'document.fields_corrected': 'Fields corrected',
  'document.approved': 'Approved',
  'document.rejected': 'Rejected',
  'document.exported': 'Exported to ERP',
  'document.failed': 'Processing failed',
  'document.retried': 'Re-queued',
};

function formatValue(value: unknown): string {
  if (value === null) return '∅';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

export function AuditTimeline({ documentId }: { documentId: string }) {
  const audit = useAudit(documentId);

  if (audit.isPending) {
    return (
      <div aria-busy="true">
        <SkeletonText lines={6} />
      </div>
    );
  }
  if (audit.isError) {
    return (
      <Callout
        tone="danger"
        title="The audit trail could not be loaded"
        actions={
          <Button size="sm" onClick={() => void audit.refetch()}>
            Retry
          </Button>
        }
      >
        {describeError(audit.error)}
      </Callout>
    );
  }
  if (audit.data.length === 0) {
    return <p className={styles.muted}>No events recorded yet.</p>;
  }

  const events = [...audit.data].sort((a, b) => b.created_at.localeCompare(a.created_at));
  return (
    <ol className={styles.timeline} aria-label="Audit trail, newest first">
      {events.map((event) => (
        <AuditEntry key={event.id} event={event} />
      ))}
    </ol>
  );
}

function AuditEntry({ event }: { event: AuditEvent }) {
  const diff = event.diff ? Object.entries(event.diff) : [];
  return (
    <li className={styles.entry}>
      <div className={styles.headline}>
        <span className={styles.action}>{ACTION_LABEL[event.action] ?? event.action}</span>
        <span className={styles.muted}>by {event.actor?.name ?? 'System'}</span>
      </div>
      <time dateTime={event.created_at} className={styles.muted}>
        {formatDateTime(event.created_at)}
      </time>
      {diff.length > 0 && (
        <table className={styles.diff}>
          <caption className="visually-hidden">Changed fields</caption>
          <thead>
            <tr>
              <th scope="col">Field</th>
              <th scope="col">Before</th>
              <th scope="col">After</th>
            </tr>
          </thead>
          <tbody>
            {diff.map(([field, change]) => (
              <tr key={field}>
                <th scope="row">{field}</th>
                <td className={styles.before}>{formatValue(change.before)}</td>
                <td className={styles.after}>{formatValue(change.after)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {event.request_id && <code className={styles.requestId}>request {event.request_id}</code>}
    </li>
  );
}
