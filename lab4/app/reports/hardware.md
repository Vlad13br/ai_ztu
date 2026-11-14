## 1.1. Залізо

| Параметр | Значення | Звідки взято |
|---|---|---|
| Компʼютер | MacBook Air (Mac17,4) | system_profiler SPHardwareDataType, sysctl hw.model |
| macOS | 26.6.2 (arm64) | sw_vers, uname -m |
| Процесор | Apple M5, 10 ядер (4 продуктивних + 6 енергоефективних) | sysctl machdep.cpu.brand_string, hw.physicalcpu, hw.perflevel* |
| Оперативна памʼять (RAM) | 24 ГБ уніфікованої памʼяті (спільна для CPU і GPU) | sysctl hw.memsize |
| Відеокарта | Apple M5 (вбудована в чип), ядер GPU: 10 | system_profiler SPDisplaysDataType |
| «VRAM» (скільки RAM доступно GPU) | ≈ 16.1 ГБ | оцінка: macOS за замовчуванням дає GPU ≈67% RAM; точне число зʼявиться в логі Ollama після першого запуску моделі - перезапустіть скрипт |
| Пропускна здатність памʼяті | 153 ГБ/с | специфікація Apple для Apple M5 (apple.com → Tech Specs) |
