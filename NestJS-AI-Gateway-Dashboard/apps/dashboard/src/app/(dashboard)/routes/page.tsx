import type { Metadata } from 'next';
import type { CircuitState } from '@aigw/shared';
import { PageHeader } from '@/components/PageHeader';
import ui from '@/components/ui.module.css';
import { withAdmin } from '@/lib/admin';
import styles from './routes.module.css';

export const metadata: Metadata = { title: 'Models & routes' };

const CIRCUIT_LABEL: Record<CircuitState, string> = {
  closed: 'Healthy',
  half_open: 'Probing',
  open: 'Circuit open',
};

function formatPrice(value: number | null): string {
  return value === null ? '—' : `$${value.toFixed(2)}`;
}

export default async function RoutesPage() {
  const routes = await withAdmin((admin) => admin.listRoutes());

  return (
    <>
      <PageHeader title="Models & routes">
        Clients request an alias; the gateway tries its targets in order and falls back on rate
        limits, server errors and timeouts.
      </PageHeader>

      {routes.map((route) => (
        <section key={route.alias} className={ui.panel} aria-labelledby={`route-${route.alias}`}>
          <h2 id={`route-${route.alias}`} className={ui.panelTitle}>
            <code>{route.alias}</code>
          </h2>
          <p className={ui.panelHint}>{route.description}</p>
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th scope="col">Priority</th>
                  <th scope="col">Provider</th>
                  <th scope="col">Model</th>
                  <th scope="col" className={ui.num}>
                    Input / 1M tokens
                  </th>
                  <th scope="col" className={ui.num}>
                    Output / 1M tokens
                  </th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {route.targets.map((target, index) => (
                  <tr key={`${target.provider}/${target.model}`}>
                    <td className={styles.priority}>
                      {index === 0 ? 'Primary' : `Fallback ${index}`}
                    </td>
                    <td>{target.provider}</td>
                    <td className={ui.mono}>{target.model}</td>
                    <td className={ui.num}>{formatPrice(target.inputPerMillionUsd)}</td>
                    <td className={ui.num}>{formatPrice(target.outputPerMillionUsd)}</td>
                    <td>
                      {!target.enabled ? (
                        <span className={ui.badge}>Not configured</span>
                      ) : target.circuit === 'closed' ? (
                        <span className={ui.badgeGood}>
                          <span aria-hidden="true">●</span> {CIRCUIT_LABEL.closed}
                        </span>
                      ) : (
                        <span className={ui.badgeCritical}>
                          <span aria-hidden="true">▲</span> {CIRCUIT_LABEL[target.circuit]}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </>
  );
}
