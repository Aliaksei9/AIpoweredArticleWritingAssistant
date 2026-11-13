export function forecastInput(localPublication) {
  const date = localPublication ? new Date(localPublication) : null;
  if (date && !Number.isFinite(date.getTime())) throw new Error("Проверьте время публикации.");
  return { published_at: date ? date.toISOString() : null };
}

export function forecastCaption(context) {
  if (!context?.horizon_modelled) return null;
  return "Через семь дней после публикации · ориентировочная оценка.";
}

export function publicationCaption(localPublication) {
  const { published_at } = forecastInput(localPublication);
  if (!published_at) return "Сейчас";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit"
  }).format(new Date(published_at));
}
