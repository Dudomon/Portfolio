import styles from './StatCard.module.css';

interface StatCardProps {
  label: string;
  value: string;
  /** Optional context line, e.g. the comparison basis. */
  detail?: string;
}

/** A headline number. Used instead of a chart when the job is "how much, right now". */
export function StatCard({ label, value, detail }: StatCardProps) {
  return (
    <div className={styles.card}>
      <dt className={styles.label}>{label}</dt>
      <dd className={styles.value}>{value}</dd>
      {detail ? <dd className={styles.detail}>{detail}</dd> : null}
    </div>
  );
}
