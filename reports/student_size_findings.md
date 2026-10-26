## Findings

1. **A bigger student helps more than a fancier teacher.** Without any teacher, going from 12 k to
   45 k parameters lifts specificity at 95 % sensitivity from 88.2 % to 92.5 %. The best student
   overall is the 94 k one at **94.8 %** (12 k today: 91.0 % with the ResNet teacher, 91.7 % at best).
2. **Distillation still helps at every size.** The best distilled student beats the same size
   without a teacher by +3.5, +3.8, +1.9 and +2.6 points (12 k → 94 k), and cuts false alarms on
   the elderly (e.g. 12 k: 43.2 → 14.8 per hour; 45 k: 11.0 → 8.9).
3. **The best teacher doesn't make the best student.** The small transformer was the *weakest*
   teacher (90.0 %) but taught the best 26 k, 45 k and 94 k students; the Oxford pretrained teacher
   only won at 12 k. Fold-to-fold spread is ±3–5 points, so among teachers at a given size most
   gaps are noise; the solid conclusions are "bigger student" and "use a teacher".
4. **45 k is the sweet spot.** 94.4 % (transformer S teacher) for 59 KB of flash, ~13 KB of RAM and
   an estimated 29–69 ms per inference. Doubling to 94 k adds only +0.4 points for twice the compute,
   and its slow-end estimate (142 ms) passes the 100 ms target.
5. **The decision threshold needs better calibration before deployment.** Thresholds picked on
   the small validation sets for 95 % sensitivity catch only ~87–89 % of test falls with the bigger
   students (90.1 % for today's 12 k). The model can do better (that's what the 95 %-sensitivity
   column shows); the threshold is being set from too few validation falls.

**Recommendation for the band:** a 45 k Sanad-Tiny distilled from the small transformer (licence-
free, no Oxford weights involved), with its decision threshold calibrated on pooled out-of-fold
predictions instead of one small validation set.

