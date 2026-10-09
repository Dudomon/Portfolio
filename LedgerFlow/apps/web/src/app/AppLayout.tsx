import { Suspense } from 'react';
import { NavLink, Outlet } from 'react-router-dom';

import { useMe } from '@/api';
import { useAuth, useSession } from '@/features/auth';
import { cx } from '@/lib/cx';
import { useTheme } from '@/lib/useTheme';
import { Badge, Button, IconButton, MoonIcon, Skeleton, SkeletonText, SunIcon } from '@/ui';

import styles from './AppLayout.module.css';

const NAV_ITEMS = [
  { to: '/documents', label: 'Documents' },
  { to: '/dashboard', label: 'Dashboard' },
] as const;

export function AppLayout() {
  const session = useSession();
  const { signOut } = useAuth();
  const me = useMe();
  const { theme, toggleTheme } = useTheme();

  return (
    <div className={styles.shell}>
      <a href="#main" className={styles.skipLink}>
        Skip to content
      </a>
      <header className={styles.topbar}>
        <div className={styles.brand}>
          <span className={styles.logo} aria-hidden="true">
            L
          </span>
          LedgerFlow
        </div>
        <nav aria-label="Main" className={styles.nav}>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => cx(styles.navLink, isActive && styles.navLinkActive)}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className={styles.user}>
          <span className={styles.userName}>{me.data?.name ?? session.claims.sub}</span>
          <Badge>{session.claims.role}</Badge>
          <IconButton
            label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            icon={theme === 'dark' ? <SunIcon /> : <MoonIcon />}
            onClick={toggleTheme}
          />
          <Button variant="ghost" size="sm" onClick={signOut}>
            Sign out
          </Button>
        </div>
      </header>
      <main id="main" className={styles.main} tabIndex={-1}>
        <Suspense fallback={<PageSkeleton />}>
          <Outlet />
        </Suspense>
      </main>
    </div>
  );
}

function PageSkeleton() {
  return (
    <div className={styles.pageSkeleton} aria-busy="true">
      <span className="visually-hidden" role="status">
        Loading page
      </span>
      <Skeleton width="12rem" height="1.75rem" />
      <SkeletonText lines={8} />
    </div>
  );
}
