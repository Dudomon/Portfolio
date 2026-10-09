import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { Link, useBlocker, useNavigate, useParams } from 'react-router-dom';

import {
  ApiError,
  PreconditionFailedError,
  describeError,
  useApprove,
  useCorrectFields,
  useDocument,
  useReject,
  useRetryDocument,
  type DocumentDetail,
  type DocumentStatus,
  type Versioned,
} from '@/api';
import { useHotkeys } from '@/lib/useHotkeys';
import { useMediaQuery } from '@/lib/useMediaQuery';
import {
  Button,
  Callout,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ConfidenceMeter,
  Dialog,
  IconButton,
  Skeleton,
  SkeletonText,
  Tab,
  TabList,
  TabPanel,
  Tabs,
  useToast,
} from '@/ui';

import { RequireRole, useHasRole } from '../auth';
import { StatusBadge } from '../documents';
import { AuditTimeline } from './AuditTimeline';
import { ConflictDialog } from './ConflictDialog';
import { DocumentPreview } from './DocumentPreview';
import {
  CONFIDENCE_THRESHOLD,
  FIELD_GROUPS,
  createReviewSchema,
  toCorrections,
  toFormValues,
  type ReviewFormValues,
} from './fieldModel';
import { LineItemsEditor } from './LineItemsEditor';
import { RejectDialog } from './RejectDialog';
import { ReviewField } from './ReviewField';
import formStyles from './ReviewForm.module.css';
import styles from './ReviewPage.module.css';
import { useReviewQueue } from './useReviewQueue';

const REVIEWABLE: ReadonlySet<DocumentStatus> = new Set(['needs_review', 'extracted']);
const WIDE_LAYOUT = '(min-width: 1024px)';

export function ReviewPage() {
  const { documentId = '' } = useParams();
  const query = useDocument(documentId);

  if (query.isPending) return <ReviewSkeleton />;
  if (query.isError) {
    const notFound = query.error instanceof ApiError && query.error.status === 404;
    return (
      <Callout
        tone="danger"
        title={notFound ? 'Document not found' : 'The document could not be loaded'}
        actions={
          notFound ? (
            <Link to="/documents">Back to documents</Link>
          ) : (
            <Button size="sm" onClick={() => void query.refetch()}>
              Retry
            </Button>
          )
        }
      >
        {!notFound && describeError(query.error)}
      </Callout>
    );
  }
  // Keyed so that moving through the queue starts from a clean form and snapshot.
  return <ReviewWorkbench key={documentId} initial={query.data} />;
}

function ReviewWorkbench({ initial }: { initial: Versioned<DocumentDetail> }) {
  const id = initial.data.id;
  const navigate = useNavigate();
  const toast = useToast();
  const wide = useMediaQuery(WIDE_LAYOUT);
  const canReview = useHasRole('reviewer');

  const documentQuery = useDocument(id);
  // `live` follows the cache (background refetches, optimistic updates);
  // `base` is the snapshot the form and every If-Match are built from.
  const live = documentQuery.data ?? initial;
  const [base, setBase] = useState(initial);
  const document = live.data;
  const fieldMeta = document.extraction?.fields ?? {};
  const reviewable = canReview && REVIEWABLE.has(document.status);

  const schema = useMemo(() => createReviewSchema(document.doc_type), [document.doc_type]);
  const defaults = useMemo(() => toFormValues(base.data), [base]);
  const form = useForm<ReviewFormValues>({
    resolver: zodResolver(schema),
    defaultValues: defaults,
  });
  const { reset } = form;
  const { isDirty } = form.formState;
  useEffect(() => reset(defaults), [defaults, reset]);

  const correct = useCorrectFields(id);
  const approve = useApprove(id);
  const reject = useReject(id);
  const retry = useRetryDocument(id);
  const queue = useReviewQueue(id);
  const busy = correct.isPending || approve.isPending || reject.isPending;

  const [rejectOpen, setRejectOpen] = useState(false);
  const [conflictOpen, setConflictOpen] = useState(false);
  const [reloading, setReloading] = useState(false);

  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      isDirty && currentLocation.pathname !== nextLocation.pathname,
  );

  function handleMutationError(error: Error) {
    if (error instanceof PreconditionFailedError) {
      setConflictOpen(true);
      return;
    }
    toast.show({ tone: 'danger', title: 'The action failed', description: describeError(error) });
  }

  function goTo(targetId: string | undefined) {
    if (targetId) navigate(`/documents/${targetId}`);
  }

  const save = form.handleSubmit((values) => {
    const fields = toCorrections(values, defaults);
    if (Object.keys(fields).length === 0) {
      reset(defaults);
      return;
    }
    correct.mutate(
      { fields, etag: base.etag },
      {
        onSuccess: (result) => {
          setBase(result);
          toast.show({ tone: 'success', title: 'Corrections saved' });
        },
        onError: handleMutationError,
      },
    );
  });

  function approveDocument() {
    if (!reviewable || busy) return;
    if (isDirty) {
      toast.show({ tone: 'warning', title: 'Save or discard your corrections before approving.' });
      return;
    }
    const nextId = queue.nextId;
    approve.mutate(
      { etag: base.etag },
      {
        onSuccess: (result) => {
          setBase(result);
          toast.show({
            tone: 'success',
            title: 'Document approved',
            description: document.filename,
          });
          goTo(nextId);
        },
        onError: handleMutationError,
      },
    );
  }

  function rejectDocument(reason: string) {
    const nextId = queue.nextId;
    reject.mutate(
      { reason, etag: base.etag },
      {
        onSuccess: (result) => {
          setBase(result);
          setRejectOpen(false);
          toast.show({
            tone: 'success',
            title: 'Document rejected',
            description: document.filename,
          });
          goTo(nextId);
        },
        onError: (error) => {
          setRejectOpen(false);
          handleMutationError(error);
        },
      },
    );
  }

  async function reloadLatest() {
    setReloading(true);
    const result = await documentQuery.refetch();
    setReloading(false);
    if (result.data) {
      setBase(result.data);
      setConflictOpen(false);
    }
  }

  useHotkeys({
    a: approveDocument,
    r: () => {
      if (reviewable && !busy) setRejectOpen(true);
    },
    j: () => goTo(queue.nextId),
    k: () => goTo(queue.previousId),
  });

  const lowConfidenceCount = Object.values(fieldMeta).filter(
    (field) => field.confidence < CONFIDENCE_THRESHOLD,
  ).length;
  const showNfeKey = document.doc_type === 'nfe' || defaults.nfe_access_key !== '';

  const fieldsPanel = document.extraction ? (
    <FormProvider {...form}>
      <form className={formStyles.form} onSubmit={save} noValidate aria-label="Extracted fields">
        {lowConfidenceCount > 0 && reviewable && (
          <Callout
            tone="warning"
            title={`${lowConfidenceCount} field(s) below ${CONFIDENCE_THRESHOLD * 100}% confidence`}
          >
            Highlighted fields need a check against the original.
          </Callout>
        )}
        {FIELD_GROUPS.map((group) => (
          <fieldset key={group.title} className={formStyles.group}>
            <legend className={formStyles.groupTitle}>{group.title}</legend>
            <div className={formStyles.grid}>
              {group.fields
                .filter((definition) => !definition.nfeOnly || showNfeKey)
                .map((definition) => (
                  <ReviewField
                    key={definition.key}
                    definition={definition}
                    meta={fieldMeta[definition.key]}
                    readOnly={!reviewable}
                  />
                ))}
            </div>
          </fieldset>
        ))}
        <LineItemsEditor meta={fieldMeta.line_items} readOnly={!reviewable} />
        {reviewable && (
          <div className={styles.formActions}>
            <Button onClick={() => reset(defaults)} disabled={!isDirty || busy}>
              Discard changes
            </Button>
            <Button type="submit" variant="primary" disabled={!isDirty} loading={correct.isPending}>
              Save corrections
            </Button>
          </div>
        )}
      </form>
    </FormProvider>
  ) : (
    <Callout title="No extraction yet">
      Fields appear here once the worker has processed this document.
    </Callout>
  );

  const auditPanel = <AuditTimeline documentId={id} />;
  const preview = <DocumentPreview document={document} />;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.titleBlock}>
          <Link to="/documents" className={styles.back}>
            <ChevronLeftIcon /> Documents
          </Link>
          <h1 className={styles.title}>{document.filename}</h1>
          <div className={styles.subtitle}>
            <StatusBadge status={document.status} />
            <ConfidenceMeter value={document.overall_confidence} label="Overall confidence" />
            {queue.position !== undefined && (
              <span className={styles.muted}>
                Queue {queue.position} of {queue.loadedCount}
              </span>
            )}
          </div>
        </div>

        <div className={styles.actions}>
          <IconButton
            label="Previous in queue"
            icon={<ChevronLeftIcon />}
            variant="secondary"
            aria-keyshortcuts="k"
            disabled={!queue.previousId}
            onClick={() => goTo(queue.previousId)}
          />
          <IconButton
            label="Next in queue"
            icon={<ChevronRightIcon />}
            variant="secondary"
            aria-keyshortcuts="j"
            disabled={!queue.nextId}
            onClick={() => goTo(queue.nextId)}
          />
          {document.status === 'failed' && (
            <RequireRole minRole="admin">
              <Button
                loading={retry.isPending}
                onClick={() =>
                  retry.mutate(undefined, {
                    onSuccess: () => toast.show({ tone: 'success', title: 'Document re-queued' }),
                    onError: handleMutationError,
                  })
                }
              >
                Re-queue
              </Button>
            </RequireRole>
          )}
          {reviewable && (
            <>
              <Button shortcut="r" disabled={busy} onClick={() => setRejectOpen(true)}>
                Reject
              </Button>
              <Button
                variant="primary"
                shortcut="a"
                startIcon={<CheckIcon />}
                disabled={isDirty || (busy && !approve.isPending)}
                loading={approve.isPending}
                onClick={approveDocument}
              >
                Approve
              </Button>
            </>
          )}
        </div>
      </header>

      {live.data.version > base.data.version && (
        <Callout
          tone="warning"
          title="This document changed since you opened it"
          actions={
            <Button size="sm" loading={reloading} onClick={() => void reloadLatest()}>
              Reload
            </Button>
          }
        >
          Saving or approving now would be rejected; reload to review the latest version.
        </Callout>
      )}
      {document.status === 'rejected' && document.rejection_reason && (
        <Callout tone="danger" title="Rejected">
          {document.rejection_reason}
        </Callout>
      )}

      {wide ? (
        <div className={styles.split}>
          <section className={styles.previewPane} aria-label="Original document">
            {preview}
          </section>
          <Tabs defaultValue="fields" className={styles.detailPane}>
            <TabList aria-label="Document details">
              <Tab value="fields">Fields</Tab>
              <Tab value="audit">Audit trail</Tab>
            </TabList>
            <TabPanel value="fields" keepMounted className={styles.panel}>
              {fieldsPanel}
            </TabPanel>
            <TabPanel value="audit" className={styles.panel}>
              {auditPanel}
            </TabPanel>
          </Tabs>
        </div>
      ) : (
        <Tabs defaultValue="fields">
          <TabList aria-label="Document views">
            <Tab value="document">Document</Tab>
            <Tab value="fields">Fields</Tab>
            <Tab value="audit">Audit trail</Tab>
          </TabList>
          <TabPanel value="document" keepMounted className={styles.panel}>
            {preview}
          </TabPanel>
          <TabPanel value="fields" keepMounted className={styles.panel}>
            {fieldsPanel}
          </TabPanel>
          <TabPanel value="audit" className={styles.panel}>
            {auditPanel}
          </TabPanel>
        </Tabs>
      )}

      <p className={styles.shortcuts}>
        Shortcuts: <Kbd>A</Kbd> approve · <Kbd>R</Kbd> reject · <Kbd>J</Kbd> / <Kbd>K</Kbd> next /
        previous in queue
      </p>

      <RejectDialog
        open={rejectOpen}
        pending={reject.isPending}
        onCancel={() => setRejectOpen(false)}
        onConfirm={rejectDocument}
      />
      <ConflictDialog
        open={conflictOpen}
        reloading={reloading}
        onReload={() => void reloadLatest()}
        onDismiss={() => setConflictOpen(false)}
      />
      <Dialog
        open={blocker.state === 'blocked'}
        onClose={() => blocker.reset?.()}
        role="alertdialog"
        size="sm"
        title="Discard unsaved corrections?"
        description="You have edits on this document that have not been saved."
        footer={
          <>
            <Button onClick={() => blocker.reset?.()}>Keep editing</Button>
            <Button variant="danger" onClick={() => blocker.proceed?.()}>
              Discard
            </Button>
          </>
        }
      />
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return <kbd className={styles.kbd}>{children}</kbd>;
}

function ReviewSkeleton() {
  return (
    <div className={styles.page} aria-busy="true">
      <span className="visually-hidden" role="status">
        Loading document
      </span>
      <Skeleton width="40%" height="1.75rem" />
      <div className={styles.split}>
        <Skeleton shape="rect" height="32rem" />
        <SkeletonText lines={10} />
      </div>
    </div>
  );
}
