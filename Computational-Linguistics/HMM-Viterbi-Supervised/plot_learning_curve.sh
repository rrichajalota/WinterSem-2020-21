for i in 0.0 0.2 0.4 0.6 0.8 1.0
do
  python main.py  --smoothing_factor $i --tagged_file_path outputs/smooth-${i}_de-tagged.tt
  python eval.py de-utb/de-eval.tt outputs/smooth-${i}_de-tagged.tt > outputs/learning_curve/smoothing/${i}.txt
  sleep 60
done
python plot_curve_from_outputs.py