import type { ReactNode } from 'react';
import { NavLinks } from '@/components/NavLinks';
import styles from '@/components/shell.module.css';
import ui from '@/components/ui.module.css';
import { logout } from '../login/actions';

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className={styles.shell}>
      <nav className={styles.sidebar} aria-label="Main">
        <span className={styles.brand}>AI Gateway</span>
        <NavLinks />
        <form action={logout} className={styles.logout}>
          <button type="submit" className={ui.buttonSecondary}>
            Sign out
          </button>
        </form>
      </nav>
      <main className={styles.main}>{children}</main>
    </div>
  );
}
