# Сторонние ассеты интерфейса

Все файлы лежат в репозитории и отдаются самим сервером: интерфейс не ходит в CDN и работает в закрытом контуре.
JavaScript-библиотек нет: весь код в `web/` написан командой.

| Ассет | Что это | Лицензия | Источник |
|---|---|---|---|
| `fonts/onest-*-wght-normal.woff2` | шрифт интерфейса Onest (variable, wght 100–900), подмножества cyrillic, cyrillic-ext, latin, latin-ext | SIL OFL 1.1, © The Onest Project Authors, текст: [licenses/OFL-Onest.txt](licenses/OFL-Onest.txt) | github.com/simpals/onest, файлы из npm-пакета `@fontsource-variable/onest@5.3.1` |
| `fonts/literata-*-wght-normal.woff2` | шрифт для текста карточки Literata (variable, wght 200–900), те же подмножества | SIL OFL 1.1, © The Literata Project Authors, текст: [licenses/OFL-Literata.txt](licenses/OFL-Literata.txt) | github.com/googlefonts/literata, файлы из npm-пакета `@fontsource-variable/literata@5.3.0` |
| `icons.svg` | спрайт из 53 иконок Phosphor (начертание regular), собран из отдельных SVG без изменений путей | MIT, © 2023 Phosphor Icons, текст: [licenses/MIT-Phosphor.txt](licenses/MIT-Phosphor.txt) | phosphoricons.com, npm-пакет `@phosphor-icons/core@2.1.1` |
| `favicon.svg` | знак attrx: наклонная плашка | собственная работа | — |

Фирменный красный М.Видео `#E2231A` снят пипеткой с презентации кейса (`data/raw/case_brief.pdf`, акцентные элементы
слайдов). Логотип М.Видео в интерфейсе не используется.

Палитра графиков (синий `#2a78d6`, бирюзовый `#0f8a63`, красный рабочей точки) проверена валидатором цветовых палитр
из навыка dataviz (различимость при дальтонизме, контраст к фону в светлой и тёмной теме).
