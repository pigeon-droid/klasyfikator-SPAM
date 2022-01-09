## Raport z projektu

### Temat - SPAM czy nie?
---
Napisałem klasyfikator maili korzystając z regresji logitycznej.
Projekt powstał w Pythonie z wykorzystaniem bibliotek:
- spacy (do tokenizacji maili);
- sklearn (do stworzenia modelu regresji);
- pandas i numpy (do zarządzania danymi);
- matplotlib (do wizualizacji danych);
- pickle (do zapisania modelu)

Maile do projektu zostały [pobrane z seriwsu Kaggle](https://www.kaggle.com/karthickveerakumar/spam-filter) i znajdują się nadesłanym rozwiązaniu. Z racji tego, że jest zbiór maili w języku angielskim, klasyfikator będzie działał wyłącznie dla maili w tym języku.

Program wydaje się na tyle prosty, że pozostawiam go bez wyjaśniania, licząc, że wystarczą do tego komentarze w kodzie. Należy jednak zwrócić uwagę, że przetwarzanie danych zajmuje około 10-15 minut.

Otrzymane klasyfikatorem wyniki okazały się bardzo obiecujące. Dla 5155 maili w części treningowej i 573 w części testowej otrzymałem:
- Accuracy 98.78%
- Precision 96.15%
- Recall 98.43%

Cieszą one tym bardziej, że baza maili nie była symetryczna, a więc istniało ryzyko stronniczości w kierunku bycia nie-spamem.
Wyniki mogą się oczywiście różnić pomiędzy dwoma uruchomieniami programu, ze względu na stochastyczność modelu i podziału danych.

Do rozwiązania załączam również wykres wizualizujący dane użyte do stworzenia modelu regresji. W mojej ocenie idealnie prezentują podział maili w zależności od wartości funkcji tfidf dla poszczególnych korpusów.
