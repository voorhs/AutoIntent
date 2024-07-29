# AutoIntent

![](assets/classification_pipeline.png)

Планы:
- протестировать все модули
- реализовать метрики для ScorerModule, PredictionModule, RegExpModule
- написать и протестить оптимизацию всего пайплайна
- подумать над кешированием запросов к collection (ибо на оптимизации k для knn и dncc можно переиспользовать много запросов)
- идея для метрики для RegExp:
    - аккураси и покрытие (эти метрики помогут понять, нужно ли вообще использовать модуль RegExp)
- подумать над проблемой переобучения: следующие этапы оптимизации должны использовать другие данные нежели предыдущие

backlog:
- optuna
- medium results caching
- logging