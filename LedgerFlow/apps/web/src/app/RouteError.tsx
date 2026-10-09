import { isRouteErrorResponse, Link, useRouteError } from 'react-router-dom';

import { describeError } from '@/api';
import { Button, Callout } from '@/ui';

import styles from './RouteError.module.css';

/**
 * Route-level error boundary. Because every route declares one, a crash in a
 * page leaves the app shell (navigation, session) usable.
 */
export function RouteError() {
  const error = useRouteError();
  const notFound = isRouteErrorResponse(error) && error.status === 404;

  return (
    <div className={styles.container}>
      <Callout
        tone="danger"
        title={notFound ? 'Page not found' : 'This page ran into a problem'}
        actions={
          <>
            <Button size="sm" onClick={() => window.location.reload()}>
              Reload
            </Button>
            <Link to="/documents" className={styles.link}>
              Go to documents
            </Link>
          </>
        }
      >
        {!notFound && describeError(error)}
      </Callout>
    </div>
  );
}

export function NotFoundPage() {
  return (
    <div className={styles.container}>
      <Callout
        title="Page not found"
        actions={
          <Link to="/documents" className={styles.link}>
            Go to documents
          </Link>
        }
      >
        The address may be mistyped, or the page has moved.
      </Callout>
    </div>
  );
}
