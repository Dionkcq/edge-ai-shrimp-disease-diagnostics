import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'
import {
  fixtureAdvice,
  fixtureGuidance,
  fixtureMeta,
  fixtureResult,
} from '../tests/support/fixtures'

async function mockLocalApi(page: Page) {
  await page.route('**/api/v1/meta', (route) => route.fulfill({ json: fixtureMeta() }))
  await page.route('**/api/v1/screenings**', (route) =>
    route.fulfill({ json: fixtureResult('MULTIPLE_TARGET_MARKERS_DETECTED') }),
  )
  await page.route('**/api/v1/guidance/**', (route) => route.fulfill({ json: fixtureGuidance() }))
  await page.route('**/api/v1/advice/**', (route) => route.fulfill({ json: fixtureAdvice() }))
}

const shrimpImage = 'e2e/fixtures/test-shrimp.jpg'
const qualityPassImage = 'e2e/fixtures/quality-pass.jpg'

test.beforeEach(async ({ page }) => {
  await mockLocalApi(page)
})

test('operates without horizontal overflow and exposes a safe result', async ({
  page,
}, testInfo) => {
  await page.goto('/')
  await expect(page.getByText('DEMO FIXTURE — NOT MODEL OUTPUT').first()).toBeVisible()
  await expect(page.getByText(/local screening service is unavailable/i)).toHaveCount(0)
  await expect(page.getByRole('link', { name: /skip to image capture/i })).toHaveCSS(
    'clip-path',
    'inset(50%)',
  )
  await page.getByLabel(/take or choose/i).setInputFiles(shrimpImage)
  await expect(page.getByText('Selected locally — not uploaded yet')).toBeVisible()
  await page.getByRole('button', { name: /screen photograph/i }).click()

  await expect(page.getByRole('heading', { name: 'Both marker types seen' })).toBeVisible()
  await expect(page.getByText(/not a calibrated disease probability/i)).toBeVisible()
  await expect(page.getByRole('heading', { name: fixtureGuidance().headline })).toBeVisible()
  await expect(page.getByRole('button', { name: /white spot marker 1/i })).toBeVisible()
  await expect(page.getByRole('button', { name: /gill darkening marker 2/i })).toBeVisible()

  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }))
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport)

  const stage = await page.locator('.image-stage').boundingBox()
  const marker = await page.getByRole('button', { name: /white spot marker 1/i }).boundingBox()
  expect(stage).not.toBeNull()
  expect(marker).not.toBeNull()
  if (!stage || !marker) throw new Error('evidence stage and marker must be laid out')
  expect(marker.x).toBeGreaterThanOrEqual(stage.x)
  expect(marker.y).toBeGreaterThanOrEqual(stage.y)
  expect(marker.x + marker.width).toBeLessThanOrEqual(stage.x + stage.width + 1)
  expect(marker.y + marker.height).toBeLessThanOrEqual(stage.y + stage.height + 1)

  await page.screenshot({ path: testInfo.outputPath('operate.png'), fullPage: true })
})

test('generates optional advice only on request and always with its disclosure', async ({
  page,
}) => {
  await page.goto('/')
  await page.getByLabel(/take or choose/i).setInputFiles(shrimpImage)
  await page.getByRole('button', { name: /screen photograph/i }).click()
  await expect(page.getByRole('heading', { name: 'Both marker types seen' })).toBeVisible()

  // Nothing is generated until a person asks for it.
  await expect(page.getByText(fixtureAdvice().summary)).toHaveCount(0)
  await page.getByRole('button', { name: /generate explanation/i }).click()

  await expect(page.getByText(/ai generated · not reviewed/i)).toBeVisible()
  await expect(page.getByText(fixtureAdvice().summary)).toBeVisible()
  await expect(page.getByText(fixtureAdvice().review_note)).toBeVisible()
  await expect(page.locator('.advice-provenance')).toContainText(fixtureAdvice().model_id)

  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }))
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport)
})

test('passes browser accessibility checks and keyboard focus remains visible', async ({ page }) => {
  await page.goto('/')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: /skip to image capture/i })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.locator('#operate')).toBeVisible()

  const accessibility = await new AxeBuilder({ page }).analyze()
  expect(accessibility.violations).toEqual([])
})

test('honours reduced motion and reports service unavailability honestly', async ({ page }) => {
  await page.unroute('**/api/v1/meta')
  await page.route('**/api/v1/meta', (route) => route.fulfill({ status: 503, body: '' }))
  await page.emulateMedia({ reducedMotion: 'reduce' })
  await page.goto('/')
  await expect(page.getByText(/local screening service is unavailable/i)).toBeVisible()
  await expect(page.getByLabel(/take or choose/i)).toBeEnabled()
  const transition = await page
    .locator('.button')
    .first()
    .evaluate((element) => getComputedStyle(element).transitionDuration)
  expect(transition.endsWith('s')).toBe(true)
  expect(Number.parseFloat(transition)).toBeLessThanOrEqual(0.000_01)
})

test('joins the built frontend to the real fail-closed FastAPI service', async ({ page }) => {
  await page.unrouteAll({ behavior: 'wait' })
  await page.goto('/')

  await expect(page.getByText('DEMO FIXTURE — NOT MODEL OUTPUT')).toHaveCount(0)
  const readiness = await page.request.get('/readyz')
  expect(readiness.status()).toBe(503)

  await page.getByLabel(/take or choose/i).setInputFiles(qualityPassImage)
  await page.getByRole('button', { name: /screen photograph/i }).click()

  await expect(page.getByRole('heading', { name: 'Unable to assess this image' })).toBeVisible()
  await expect(page.locator('.specific-action')).toContainText(/no screening model is installed/i)
  await expect(page.getByText('Review status', { exact: true })).toBeVisible()
})
