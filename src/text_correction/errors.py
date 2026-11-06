"""Errors safe to display without credentials or raw HTTP request headers."""


class TextCorrectionError(Exception):
    pass


class ContextBudgetError(TextCorrectionError):
    pass


class InvalidLLMResponse(TextCorrectionError):
    pass


class OpenRouterError(TextCorrectionError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def explanation_for_user(error):
    """Explain the failed stage without provider payloads or credentials."""
    if isinstance(error, ContextBudgetError):
        return "Текст и инструкции не помещаются в контекст модели. Уменьшите объём текста или увеличьте лимит контекста в настройках."
    if isinstance(error, OpenRouterError):
        if error.status_code == 429:
            return "OpenRouter ограничил частоту запросов к бесплатной модели. Повторите поиск позже."
        if error.status_code in (401, 403):
            return "OpenRouter отклонил ключ API или доступ к модели. Проверьте настройки подключения."
        if error.status_code in (408, 502, 503, 504):
            return "Провайдер модели OpenRouter временно недоступен. Повторите поиск позже."
        if "timeout" in str(error).lower():
            return "Модель OpenRouter не ответила за отведённое время. Повторите поиск позже."
        if "network" in str(error).lower():
            return "Не удалось связаться с OpenRouter. Проверьте соединение и повторите поиск."
        return "OpenRouter не смог выполнить запрос. Проверьте доступность выбранной модели."
    if isinstance(error, InvalidLLMResponse):
        message = str(error).lower()
        if "numeric" in message or "numerical" in message or "url" in message or "guarantee" in message:
            return "Модель изменила число, ссылку или важную оговорку. Ответ отклонён для сохранения фактов исходного текста."
        if "incomplete" in message:
            return "Модель не завершила ответ. Не удалось получить рекомендации для всех предложений."
        if "proofreading" in message:
            return "Не удалось подтвердить языковые исправления после повторного запроса к модели. Текст сохранён. Повторите проверку."
        return "Модель вернула некорректный ответ: нарушен формат или список исходных предложений."
    return "Не удалось подготовить рекомендации. Повторите поиск."
