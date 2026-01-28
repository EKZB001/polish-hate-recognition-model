# Eksperyment V2: Trening na zbiorze Ptaszczyńskiego

Eksperyment polegający na douczeniu (fine-tuning) modelu na zbiorze danych eksperckich.
Status: Archiwum / Ciekawostka.

## Wyniki
Model osiągnął wysoką precyzję, ale (podobnie jak wersja bazowa) ma tendencję do bycia ostrożnym przy klasyfikacji hejtu (niższy Recall).

* **Accuracy:** ~89-90%
* **F1 Score:** ~0.62
* **Specyfika:** Model bardzo dobrze rozpoznaje treści neutralne (mało False Positives), ale przepuszcza część hejtu.

## Pliki w tym folderze
* `*_v2.py` - skrypty dostosowane do struktury danych Ptaszczyńskiego.
* `confusion_matrix_v2.png` - macierz pomyłek ze zbioru testowego.

## Źródło danych i Cytowanie
Ten eksperyment wykorzystuje zbiór danych: "Expert-Annotated Dataset to Study Cyberbullying in Polish Language". Zgodnie z licencją, wymagana jest poniższa adnotacja:

```bibtex
@article{ptaszynski2023expert,
  title={Expert-Annotated Dataset to Study Cyberbullying in Polish Language},
  author={Ptaszynski, Michal and Pieciukiewicz, Agata and Dybala, Pawel and Skrzek, Pawel and Soliwoda, Kamil and Fortuna, Marcin and Leliwa, Gniewosz and Wroczynski, Michal},
  journal={Data},
  volume={9},
  number={1},
  pages={1},
  year={2023},
  publisher={MDPI}
}
