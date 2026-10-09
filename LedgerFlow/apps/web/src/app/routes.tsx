import { lazy } from 'react';
import { Navigate, type RouteObject } from 'react-router-dom';

import { LoginPage, RequireAuth } from '@/features/auth';

import { AppLayout } from './AppLayout';
import { NotFoundPage, RouteError } from './RouteError';

// Pages are split per route; AppLayout's Suspense boundary shows a skeleton while they load.
const DashboardPage = lazy(() =>
  import('@/features/dashboard').then((module) => ({ default: module.DashboardPage })),
);
const DocumentsPage = lazy(() =>
  import('@/features/documents').then((module) => ({ default: module.DocumentsPage })),
);
const ReviewPage = lazy(() =>
  import('@/features/review').then((module) => ({ default: module.ReviewPage })),
);

export const routes: RouteObject[] = [
  { path: '/login', element: <LoginPage />, errorElement: <RouteError /> },
  {
    element: <RequireAuth />,
    errorElement: <RouteError />,
    children: [
      {
        element: <AppLayout />,
        children: [
          { index: true, element: <Navigate to="/documents" replace /> },
          { path: 'dashboard', element: <DashboardPage />, errorElement: <RouteError /> },
          { path: 'documents', element: <DocumentsPage />, errorElement: <RouteError /> },
          {
            path: 'documents/:documentId',
            element: <ReviewPage />,
            errorElement: <RouteError />,
          },
          { path: '*', element: <NotFoundPage /> },
        ],
      },
    ],
  },
];
