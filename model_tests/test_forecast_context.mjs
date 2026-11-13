import test from 'node:test';
import assert from 'node:assert/strict';
import { forecastInput, forecastCaption, publicationCaption } from '../src/ui/static/forecast_context.mjs';

test('Publication plan converts local time to an explicit UTC instant', () => {
  const plan=forecastInput('2026-11-12T15:30');
  assert.equal(plan.published_at,new Date('2026-11-12T15:30').toISOString());
  assert.ok(!Object.hasOwn(plan,'horizon_days'));
  assert.deepEqual(forecastInput(''),{published_at:null});
});
test('Invalid date is rejected and old saved horizon cannot enter a request', () => {
  assert.throws(() => forecastInput('invalid'));
  assert.deepEqual(forecastInput('',14),{published_at:null});
});
test('Caption describes one shared horizon for all metrics', () => {
  const text=forecastCaption({horizon_modelled:true,horizon_days:7});
  assert.match(text,/Через семь дней после публикации/);
  assert.equal(forecastCaption({horizon_modelled:false}),null);
});

test('Publication display is readable while the request retains its exact timestamp', () => {
  assert.equal(publicationCaption(''),'Сейчас');
  const value='2026-11-13T18:45';
  assert.match(publicationCaption(value),/13 нояб/);
  assert.match(publicationCaption(value),/2026/);
  assert.match(publicationCaption(value),/18:45/);
  assert.equal(forecastInput(value).published_at,new Date(value).toISOString());
});
