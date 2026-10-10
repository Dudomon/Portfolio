'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './shell.module.css';

const LINKS = [
  { href: '/', label: 'Overview' },
  { href: '/keys', label: 'API keys' },
  { href: '/routes', label: 'Models & routes' },
  { href: '/requests', label: 'Request log' },
] as const;

export function NavLinks() {
  const pathname = usePathname();
  return (
    <ul className={styles.navList}>
      {LINKS.map(({ href, label }) => {
        const active = href === '/' ? pathname === '/' : pathname.startsWith(href);
        return (
          <li key={href}>
            <Link
              href={href}
              className={active ? styles.navLinkActive : styles.navLink}
              aria-current={active ? 'page' : undefined}
            >
              {label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}
