README – robust_yolo_preprocessing

Repozytorium zawiera kod rozwijany w ramach pracy magisterskiej dotyczącej poprawy odporności detektorów obiektów YOLO na zakłócenia obrazu poprzez zastosowanie przetwarzania wstępnego.

Głównym celem projektu jest porównanie kilku wariantów przetwarzania obrazu przed detekcją:

bazowego modelu YOLOv7 (baseline),

detekcji na obrazach zdegradowanych (np. szum, kompresja JPEG),

klasycznej filtracji obrazu (np. filtr medianowy),

podejścia z learnable preprocessing block (LPB), czyli modułu uczonego razem z detektorem.

Projekt jest obecnie na etapie budowy pipeline’u eksperymentalnego i wstępnych testów.

Stan obecny projektu

Na obecnym etapie repozytorium zawiera:

bazową implementację YOLOv7,

kod do uruchamiania testów detekcji,

skrypt benchmarkowy uruchamiający eksperymenty dla różnych wariantów danych,

narzędzia do przygotowania prostego zbioru testowego (coco128),

narzędzia do budowania wariantów danych z zakłóceniami.

Repozytorium nie zawiera samych danych ani wag modeli (są one trzymane lokalnie i ignorowane przez git).

Co działa

Obecnie działają następujące elementy:

Uruchamianie detekcji YOLOv7 na zbiorze testowym coco128.

Automatyczny skrypt benchmarkowy uruchamiający kilka wariantów eksperymentu:

clean (obrazy bez zakłóceń),

noise (obrazy z dodanym szumem),

jpeg (obrazy z kompresją JPEG),

warianty z filtracją medianową.

Zbieranie wyników z uruchomień modelu oraz zapis ich do plików CSV.

Generowanie podsumowania wyników dla poszczególnych wariantów eksperymentu.

Możliwość łatwego rozszerzania eksperymentu o nowe warianty przetwarzania obrazu.

Smoke test przeprowadzony na zbiorze coco128 pozwala sprawdzić czy pipeline eksperymentalny działa poprawnie.




Na obecnym etapie nie wszystkie elementy pipeline’u działają jeszcze w pełni stabilnie.

Najważniejsze rzeczy w trakcie:

Synchronizacja obrazów i etykiet przy generowaniu wariantów zbioru z zakłóceniami.
W części wariantów eksperymentów model nie widzi etykiet, co powoduje zerowe wartości metryk.

Dopracowanie skryptu benchmarkowego tak, aby automatycznie poprawnie parsował wyniki z logów YOLO i zapisywał je do plików wynikowych.

Uporządkowanie struktury danych i konfiguracji datasetów.

Integracja learnable preprocessing block (LPB) z architekturą YOLOv7 oraz przygotowanie eksperymentów porównawczych.

Przygotowanie eksperymentów na docelowym zbiorze CrowdHuman.

Plan dalszych prac

Najbliższe kroki w projekcie obejmują:

stabilizację pipeline’u eksperymentalnego,

poprawne generowanie wszystkich wariantów danych,

implementację i integrację learnable preprocessing block,

przeprowadzenie eksperymentów porównawczych,

analizę wpływu zakłóceń i filtracji na skuteczność detekcji.

Wyniki tych eksperymentów będą stanowiły podstawę części eksperymentalnej pracy magisterskiej.