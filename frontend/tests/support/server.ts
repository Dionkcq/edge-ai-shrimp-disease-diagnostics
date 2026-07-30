import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'
import { fixtureGuidance, fixtureMeta, fixtureResult } from './fixtures'

export const server = setupServer(
  http.get('/api/v1/meta', () => HttpResponse.json(fixtureMeta())),
  http.post('/api/v1/screenings', () => HttpResponse.json(fixtureResult())),
  http.get('/api/v1/guidance/:decision', ({ params }) =>
    HttpResponse.json({ ...fixtureGuidance(), decision: params.decision }),
  ),
)
