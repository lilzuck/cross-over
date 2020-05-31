EXP_NAME=WMT
ALPHA=0.10

mkdir -p checkpoints/${EXP_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3  python train.py data-bin/wmt16_en_de_bpe32k \
        --arch transformer_wmt_en_de --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --ddp-backend=no_c10d \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
              --lr 0.0007 --min-lr 1e-09 \
             --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 --dropout 0.1 \
              --max-tokens  3072   --save-dir checkpoints/${EXP_NAME}  \
              --update-freq 3 --no-progress-bar --log-format json --log-interval 50\
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
              --validate-interval 1 \
              --save-interval 1 --keep-last-epochs 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --alpha ${ALPHA} 1>logs/${EXP_NAME}.out 2>logs/${EXP_NAME}.err
python scripts/average_checkpoints.py --inputs checkpoints/${EXP_NAME} --num-epoch-checkpoints 5 --output checkpoints/${EXP_NAME}/avg5.pt

fairseq-generate data-bin/wmt16_en_de_bpe32k --path checkpoints/${EXP_NAME}/avg5.pt --batch-size 32 --beam 4 --lenpen 0.6 --remove-bpe --gen-subset test > logs/${EXP_NAME}.avg5.raw_result

GEN=logs/${EXP_NAME}.avg5.raw_result

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
python score.py --sys $SYS --ref $REF > logs/${EXP_NAME}.avg5.final_result

