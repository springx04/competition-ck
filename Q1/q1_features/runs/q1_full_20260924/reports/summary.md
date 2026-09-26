# Q1 feature extraction report

- Manifest rows: 100
- Processing: {'ok': 49, 'partial': 51}
- Pairing: {'no_issue_detected': 49, 'unverifiable': 51}
- Alignment issue sample IDs: ['-9y-fZ3swSY$_$8', '-HwX2H8Z4hY$_$2', '-HwX2H8Z4hY$_$5', '-HwX2H8Z4hY$_$6', '-HwX2H8Z4hY$_$9', '-MeTTeMJBNc$_$0', '-MeTTeMJBNc$_$7', '-NFrJFQijFE$_$1', '-NFrJFQijFE$_$2', '-RfYyzHpjk4$_$2', '-THoVjtIkeU$_$6', '-UuX1xuaiiE$_$0', '-UuX1xuaiiE$_$1', '-UuX1xuaiiE$_$3', '-UuX1xuaiiE$_$6', '-aqamKhZ1Ec$_$0', '-dxfTGcXJoc$_$1', '-dxfTGcXJoc$_$6', '-hnBHBN8p5A$_$6', '-hnBHBN8p5A$_$7', '-iRBcNs9oI8$_$3', '-iRBcNs9oI8$_$6', '-iRBcNs9oI8$_$7', '-iRBcNs9oI8$_$8', '-iRBcNs9oI8$_$9', '-lzEya4AM_4$_$6', '-mJ2ud6oKI8$_$1', '-mJ2ud6oKI8$_$2', '-mJ2ud6oKI8$_$6', '-mJ2ud6oKI8$_$8', '-mJ2ud6oKI8$_$9', '-ri04Z7vwnc$_$0', '-ri04Z7vwnc$_$2', '-ri04Z7vwnc$_$5', '-s9qJ7ATP7w$_$0', '-s9qJ7ATP7w$_$1', '-s9qJ7ATP7w$_$4', '-s9qJ7ATP7w$_$7', '-s9qJ7ATP7w$_$8', '-tANM6ETl_M$_$3', '-wMB_hJL-3o$_$7', '-wny0OAz3g8$_$0', '-wny0OAz3g8$_$1', '-wny0OAz3g8$_$2', '-wny0OAz3g8$_$3', '-wny0OAz3g8$_$5', '-wny0OAz3g8$_$7', '-wny0OAz3g8$_$9', '-yRb-Jum7EQ$_$1', '-yRb-Jum7EQ$_$5', '-yRb-Jum7EQ$_$6']
- Unresolved/quarantined sample IDs: ['-9y-fZ3swSY$_$8', '-HwX2H8Z4hY$_$2', '-HwX2H8Z4hY$_$5', '-HwX2H8Z4hY$_$6', '-HwX2H8Z4hY$_$9', '-MeTTeMJBNc$_$0', '-MeTTeMJBNc$_$7', '-NFrJFQijFE$_$1', '-NFrJFQijFE$_$2', '-RfYyzHpjk4$_$2', '-THoVjtIkeU$_$6', '-UuX1xuaiiE$_$0', '-UuX1xuaiiE$_$1', '-UuX1xuaiiE$_$3', '-UuX1xuaiiE$_$6', '-aqamKhZ1Ec$_$0', '-dxfTGcXJoc$_$1', '-dxfTGcXJoc$_$6', '-hnBHBN8p5A$_$6', '-hnBHBN8p5A$_$7', '-iRBcNs9oI8$_$3', '-iRBcNs9oI8$_$6', '-iRBcNs9oI8$_$7', '-iRBcNs9oI8$_$8', '-iRBcNs9oI8$_$9', '-lzEya4AM_4$_$6', '-mJ2ud6oKI8$_$1', '-mJ2ud6oKI8$_$2', '-mJ2ud6oKI8$_$6', '-mJ2ud6oKI8$_$8', '-mJ2ud6oKI8$_$9', '-ri04Z7vwnc$_$0', '-ri04Z7vwnc$_$2', '-ri04Z7vwnc$_$5', '-s9qJ7ATP7w$_$0', '-s9qJ7ATP7w$_$1', '-s9qJ7ATP7w$_$4', '-s9qJ7ATP7w$_$7', '-s9qJ7ATP7w$_$8', '-tANM6ETl_M$_$3', '-wMB_hJL-3o$_$7', '-wny0OAz3g8$_$0', '-wny0OAz3g8$_$1', '-wny0OAz3g8$_$2', '-wny0OAz3g8$_$3', '-wny0OAz3g8$_$5', '-wny0OAz3g8$_$7', '-wny0OAz3g8$_$9', '-yRb-Jum7EQ$_$1', '-yRb-Jum7EQ$_$5', '-yRb-Jum7EQ$_$6']
- Accepted spoken words: 1793/1926 (93.0945%)
- Eligible visual rows / selected source frames: 11646/19564 (59.5277%)
- Short/median/long review sample IDs: ['-mJ2ud6oKI8$_$6', '-wny0OAz3g8$_$2', '-yRb-Jum7EQ$_$1']
- Compact-candidate bytes: 1546572809
- Package category bytes: {'q1_compact_candidate': 1546572809, 'external_runtime': 5865966928, 'server_working_material': 6368495087, 'q2_q3_reserved': 0}
- Maximum 50-bin width: 0.5853333333333346 seconds
- Maximum recorded process peak RSS: 1676840 kB
- Maximum recorded PyTorch CUDA allocation: 957380096 bytes
- Submission weight policy: unresolved
- Human audiovisual review: only rows explicitly present in alignment_review.csv are treated as reviewed.
- Unreviewed automatic alarms remain suspected/unverifiable and paired_use=false; no automated diagnostic is reported as human confirmation.
- Runtime: Python=3.10.21 (main, Aug 27 2026, 14:42:07) [GCC 14.3.0]; platform=Linux-5.15.0-191-generic-x86_64-with-glibc2.35; GPU=NVIDIA GeForce RTX 4090, 580.178.04, 24564 MiB
- Pretrained BERT/Conformer weights and server working material are excluded from the compact-candidate count; this is not a claim that the complete competition submission is below 50,000,000 bytes.

## Modality availability

- text observed windows / valid windows: 3128/5000 (62.5600%)
- audio observed windows / valid windows: 4975/5000 (99.5000%)
- vision observed windows / valid windows: 2873/5000 (57.4600%)

## Cross-window source-weight example

sample=-3g5yACwYnA$_$13, cross-window word_id=0, bins=[8, 9, 10, 11, 12, 13]; for bin 8: source_ids=[0], overlap_s=[0.07594345238095257], weights=overlap/sum=[1.0], first_dimension=0.77122861, stored=0.77122861.
