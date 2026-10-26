## Findings

1. **A heavy transformer did not beat the plain ResNet.** The large transformer (3.2 M parameters)
   tied the ResNet teacher (94.2 % vs 94.0 % specificity at 95 % sensitivity) with 3.4× the
   parameters; the small one (0.55 M) came last (90.0 %). With ~35 people of data, model size is
   not what limits the teacher.
2. **Pretraining on big data did.** The *same* Oxford network went from 91.3 % (trained from
   scratch on our data) to **97.4 %** when it started from weights learned on 700,000 person-days of
   UK Biobank wrist data, and its false alarms on the elderly dropped from 32.6 to 2.1 per hour.
   The ensemble of all five teachers was best overall (98.7 %, no elderly false alarms).
3. **A much better teacher makes only a slightly better student.** The 3.4-point gap between the
   Oxford and ResNet teachers shrinks to 0.8 points between their students (91.7 % vs 90.9 %), and
   the top three students are within fold-to-fold noise (± ~6 points). Sanad-Tiny's 11.8 k
   parameters are now the bottleneck, not the teacher.
4. **Where the better teacher does show: false alarms.** The Oxford-distilled student gave 9.4
   false alarms per hour of hard daily activity (ResNet-distilled: 13.7, no teacher: 16.3) and 14.8
   on the elderly (24.6 and 43.2). At the validation-picked threshold it caught slightly fewer falls
   (88.5 % vs 90.2 %), so its threshold would need re-tuning; at equal sensitivity it is better.
5. **Recipe: half labels, half teacher (`standard`, T = 2) won for 5 of 6 teachers.** Learning from
   the teacher alone (`teacher_only`) was the worst recipe for 5 of 6, and a weak teacher can hurt:
   the from-scratch Oxford teacher with `teacher_only` (84.4 %) did worse than no teacher (88.2 %).

**Answer to "train a heavy model, then distil it":** a bigger architecture alone doesn't help; a
teacher that has *seen far more data* does, and it mainly cuts false alarms. To turn more of the
teacher's advantage into the band, the next levers are a somewhat larger student (Pico 2 W has room
for 25–50 k parameters) and real recordings from the band.

**Licence note:** the winning student is distilled from Oxford's weights, so it inherits their
Academic Use Licence (fine for the capstone with credit; not for a product for sale). The
ResNet-distilled student currently exported to `models/` has no such restriction.

