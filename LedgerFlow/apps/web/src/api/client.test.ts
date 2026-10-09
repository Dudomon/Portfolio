import { http, HttpResponse } from 'msw';
import { z } from 'zod';

import { server } from '@/mocks/server';

import { createApiClient } from './client';
import {
  ApiError,
  NetworkError,
  PreconditionFailedError,
  ResponseValidationError,
  UnauthorizedError,
} from './errors';

const BASE = 'http://localhost/api/v1';
const Thing = z.object({ id: z.string(), version: z.number() });

function setup(token: string | null = 'token-123') {
  const onUnauthorized = vi.fn();
  const client = createApiClient({ baseUrl: BASE, getAccessToken: () => token, onUnauthorized });
  return { client, onUnauthorized };
}

const problemResponse = (status: number, body: Record<string, unknown>) =>
  HttpResponse.json(
    { type: 'about:blank', status, ...body },
    { status, headers: { 'Content-Type': 'application/problem+json' } },
  );

describe('api client', () => {
  it('sends auth, request id and If-Match headers and returns the ETag', async () => {
    let seen: Headers | undefined;
    server.use(
      http.patch(`${BASE}/things/1`, ({ request }) => {
        seen = request.headers;
        return HttpResponse.json(
          { id: '1', version: 3 },
          { headers: { ETag: 'W/"3"', 'X-Request-ID': 'srv-1' } },
        );
      }),
    );
    const { client } = setup();

    const response = await client.request('/things/1', Thing, {
      method: 'PATCH',
      json: { name: 'x' },
      ifMatch: 'W/"2"',
    });

    expect(response).toMatchObject({
      data: { id: '1', version: 3 },
      etag: 'W/"3"',
      requestId: 'srv-1',
    });
    expect(seen?.get('Authorization')).toBe('Bearer token-123');
    expect(seen?.get('If-Match')).toBe('W/"2"');
    expect(seen?.get('Content-Type')).toBe('application/json');
    expect(seen?.get('X-Request-ID')).toMatch(/^[0-9a-f-]{36}$/);
  });

  it('serialises query parameters, skipping empty values', async () => {
    let url: URL | undefined;
    server.use(
      http.get(`${BASE}/things`, ({ request }) => {
        url = new URL(request.url);
        return HttpResponse.json({ id: '1', version: 1 });
      }),
    );
    await setup().client.request('/things', Thing, {
      query: { status: 'needs_review', q: '', cursor: undefined, limit: 50 },
    });
    expect(url?.search).toBe('?status=needs_review&limit=50');
  });

  it('maps RFC 9457 problem documents to ApiError', async () => {
    server.use(
      http.get(`${BASE}/things/404`, () =>
        problemResponse(404, {
          title: 'Document not found',
          detail: 'No such id',
          request_id: 'req-9',
        }),
      ),
    );
    const error = await setup()
      .client.request('/things/404', Thing)
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 404,
      message: 'No such id',
      requestId: 'req-9',
      problem: { title: 'Document not found' },
    });
  });

  it('surfaces 412 as a distinct PreconditionFailedError', async () => {
    server.use(
      http.post(`${BASE}/things/1/approve`, () =>
        problemResponse(412, { title: 'Precondition failed' }),
      ),
    );
    const error = await setup()
      .client.request('/things/1/approve', Thing, { method: 'POST', ifMatch: 'W/"1"' })
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(PreconditionFailedError);
    expect(error).toBeInstanceOf(ApiError);
  });

  it('notifies the session on 401 for authenticated calls only', async () => {
    server.use(
      http.get(`${BASE}/things/1`, () => problemResponse(401, { title: 'Not authenticated' })),
    );
    const { client, onUnauthorized } = setup();

    await expect(client.request('/things/1', Thing)).rejects.toBeInstanceOf(UnauthorizedError);
    expect(onUnauthorized).toHaveBeenCalledTimes(1);

    await expect(client.request('/things/1', Thing, { anonymous: true })).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    expect(onUnauthorized).toHaveBeenCalledTimes(1);
  });

  it('normalises non-problem error bodies', async () => {
    server.use(
      http.get(
        `${BASE}/things/1`,
        () =>
          new HttpResponse('<html>Bad gateway</html>', {
            status: 502,
            statusText: 'Bad Gateway',
            headers: { 'Content-Type': 'text/html' },
          }),
      ),
    );
    const error = await setup()
      .client.request('/things/1', Thing)
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 502,
      problem: { title: 'Bad Gateway', type: 'about:blank' },
    });
  });

  it('rejects 2xx bodies that violate the contract', async () => {
    server.use(http.get(`${BASE}/things/1`, () => HttpResponse.json({ id: 1 })));
    const error = await setup()
      .client.request('/things/1', Thing)
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(ResponseValidationError);
    expect((error as ResponseValidationError).issues.map((issue) => issue.path.join('.'))).toEqual([
      'id',
      'version',
    ]);
  });

  it('wraps transport failures in NetworkError', async () => {
    server.use(http.get(`${BASE}/things/1`, () => HttpResponse.error()));
    await expect(setup().client.request('/things/1', Thing)).rejects.toBeInstanceOf(NetworkError);
  });
});
