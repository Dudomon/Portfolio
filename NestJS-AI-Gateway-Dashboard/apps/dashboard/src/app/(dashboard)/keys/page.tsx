import type { Metadata } from 'next';
import { CreateKeyForm } from '@/components/CreateKeyForm';
import { PageHeader } from '@/components/PageHeader';
import { RevokeKeyButton } from '@/components/RevokeKeyButton';
import ui from '@/components/ui.module.css';
import { withAdmin } from '@/lib/admin';
import { formatTimestamp, formatUsd } from '@/lib/format';
import { createKey, revokeKey } from './actions';

export const metadata: Metadata = { title: 'API keys' };

export default async function KeysPage() {
  const keys = await withAdmin((admin) => admin.listKeys());

  return (
    <>
      <PageHeader title="API keys">
        Keys authenticate clients to <code>/v1</code>. Budgets reset on the first of each month
        (UTC).
      </PageHeader>

      <section className={ui.panel} aria-labelledby="create-title">
        <h2 id="create-title" className={ui.panelTitle}>
          Create a key
        </h2>
        <p className={ui.panelHint}>The project is created on first use.</p>
        <CreateKeyForm action={createKey} />
      </section>

      <div className={ui.tableWrap}>
        <table className={ui.table}>
          <caption>{keys.length} keys, active first</caption>
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Project</th>
              <th scope="col">Prefix</th>
              <th scope="col" className={ui.num}>
                Spend this month
              </th>
              <th scope="col" className={ui.num}>
                Budget
              </th>
              <th scope="col" className={ui.num}>
                Limit / min
              </th>
              <th scope="col">Created</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            {keys.length === 0 ? (
              <tr>
                <td colSpan={8} className={ui.empty}>
                  No keys yet. Create one above.
                </td>
              </tr>
            ) : (
              keys.map((key) => (
                <tr key={key.id}>
                  <td>{key.name}</td>
                  <td>{key.project}</td>
                  <td className={ui.mono}>gw_{key.prefix}_…</td>
                  <td className={ui.num}>{formatUsd(key.monthToDateSpendUsd)}</td>
                  <td className={ui.num}>
                    {key.monthlyBudgetUsd === null ? (
                      <span className={ui.muted}>None</span>
                    ) : (
                      formatUsd(key.monthlyBudgetUsd)
                    )}
                  </td>
                  <td className={ui.num}>{key.rateLimitPerMinute}</td>
                  <td>{formatTimestamp(key.createdAt)}</td>
                  <td>
                    {key.revokedAt ? (
                      <span className={ui.badge}>Revoked</span>
                    ) : (
                      <RevokeKeyButton keyName={key.name} onRevoke={revokeKey.bind(null, key.id)} />
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}
