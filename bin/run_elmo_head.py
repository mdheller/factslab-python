import argparse
from factslab.utility import read_data, interleave_lists, load_glove_embedding, padding
from factslab.pytorch.mlpregression import MLPTrainer
from torch.cuda import is_available
from torch import device
from os.path import expanduser
import sys
import pickle


if __name__ == "__main__":
    home = expanduser('~')
    # initialize argument parser
    description = 'Run a simple MLP with(out) attention of varying types on ELMO.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--load_data',
                        action='store_true')
    parser.add_argument('--protocol',
                        type=str,
                        default='arg')
    parser.add_argument('--datapath',
                        type=str,
                        default=home + '/Desktop/protocols/data/')
    parser.add_argument('--embeddings',
                        type=str,
                        default='/srv/models/pytorch/elmo/')
    parser.add_argument('--regressiontype',
                        type=str,
                        default="multinomial")
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--lr',
                        type=float,
                        default=0.001)
    parser.add_argument('--wd',
                        type=float,
                        default=0.0001)
    parser.add_argument('--batchsize',
                        type=int,
                        default=128)
    parser.add_argument('--layers',
                        type=str,
                        default='256,32')
    parser.add_argument('--argrep',
                        type=str,
                        default="root",
                        help='Argument representation- root, span, span-param')
    parser.add_argument('--predrep',
                        type=str,
                        default="root",
                        help='Predicate representation- root, span, span-param')
    parser.add_argument('--argcontext',
                        type=str,
                        default="none",
                        help='Argument context - none, david, param')
    parser.add_argument('--predcontext',
                        type=str,
                        default="none",
                        help='Argument context - none, david, param')
    parser.add_argument('--hand',
                        action='store_true',
                        help='Turn on hand engineering feats')
    parser.add_argument('--embed',
                        action='store_true',
                        help='Turn on elmo/glove embeddings')

    # parse arguments
    args = parser.parse_args()
    args.layers = [int(a) for a in args.layers.split(',')]
    # Dictionary storing the configuration of the model
    model_type = {'arg': {'repr': args.argrep, 'context': args.argcontext},
                  'pred': {'repr': args.predrep, 'context': args.predcontext}}

    arg_datafile = args.datapath + "arg_long_data.tsv"
    arg_attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
    arg_attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                     "abs": "Abs.Confidence"}

    pred_datafile = args.datapath + "pred_long_data.tsv"
    pred_attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
    pred_attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                      "hyp": "Hyp.Confidence"}

    all_attributes = {'arg': ['part', 'kind', 'abs'],
                      'pred': ['part', 'dyn', 'hyp']}

    if args.load_data:
        # Load the sentences
        sentences = {}
        with open(home + '/Desktop/protocols/data/sentences.tsv', 'r') as f:
            for line in f.readlines():
                id_sent = line.split('\t')
                sentences[id_sent[0]] = id_sent[1].split()

        arg_stuff = read_data(prot='arg',
                              datafile=arg_datafile,
                              attributes=all_attributes['arg'],
                              attr_map=arg_attr_map,
                              attr_conf=arg_attr_conf,
                              regressiontype=args.regressiontype,
                              sentences=sentences, batch_size=args.batchsize)
        pred_stuff = read_data(prot='pred',
                               datafile=pred_datafile,
                               attributes=all_attributes['pred'],
                               attr_map=pred_attr_map,
                               attr_conf=pred_attr_conf,
                               regressiontype=args.regressiontype,
                               sentences=sentences, batch_size=args.batchsize)

        with open("arg_train_data.pkl", "wb") as train_arg_f, open("arg_dev_data.pkl", "wb") as dev_arg_f, open("pred_train_data.pkl", "wb") as train_pred_f, open("pred_dev_data.pkl", "wb") as dev_pred_f:
            pickle.dump(arg_stuff[0], train_arg_f)
            pickle.dump(arg_stuff[1], dev_arg_f)
            pickle.dump(pred_stuff[0], train_pred_f)
            pickle.dump(pred_stuff[1], dev_pred_f)
        sys.exit(0)

    train_in = open(args.protocol + "_train_data.pkl", "rb")
    dev_in = open(args.protocol + "_dev_data.pkl", "rb")
    train_data = pickle.load(train_in)
    dev_data = pickle.load(dev_in)
    train_in.close()
    dev_in.close()

    x, y, roots, spans, context_roots, context_spans, loss_wts, hand_feats = train_data
    hand_feat_dim = len(hand_feats[0][0])
    # ELMO parameters
    if 'elmo' in args.embeddings:
        options_file = args.embeddings + "options/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = args.embeddings + "weights/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        embed_params = (options_file, weight_file)
        embed_dim = 1024
    # GlOvE parmaters
    elif 'glove' in args.embeddings:
        dev_x = dev_data[0]
        vocab = list(set([word for minib in (x + dev_x) for sent in minib for word in sent]))
        x = padding(x)
        dev_data[0] = padding(dev_data[0])
        embed_params = (load_glove_embedding(args.embeddings, vocab), vocab + ["<PAD>"])
        embed_dim = 300

    # pyTorch figures out device to do computation on
    device_to_use = device("cuda:0" if is_available() else "cpu")

    # Do ablations here before initialising

    # Initialise the model
    trainer = MLPTrainer(embed_params=embed_params,
                         all_attributes={args.protocol: all_attributes[args.protocol]}, layers=args.layers,
                         device=device_to_use, attention_type=model_type,
                         lr=args.lr, weight_decay=args.wd,
                         embedding_dim=embed_dim, hand_feat_dim=hand_feat_dim,
                         turn_on_hand_feats=args.hand,
                         turn_on_embeddings=args.embed)

    # Training phase
    trainer.fit(X=x, Y=y, loss_wts=loss_wts, roots=roots, spans=spans,
                context_roots=context_roots, context_spans=context_spans,
                hand_feats=hand_feats, dev=dev_data, epochs=args.epochs,
                prot=args.protocol)
