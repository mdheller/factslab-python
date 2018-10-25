import argparse
from factslab.utility import read_data, interleave_lists
from factslab.pytorch.mlpregression import MLPTrainer
from torch.cuda import is_available
from torch import device
from os.path import expanduser
import pickle


if __name__ == "__main__":
    home = expanduser('~')
    # initialize argument parser
    description = 'Run a simple MLP with(out) attention of varying types on ELMO.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--datapath',
                        type=str,
                        default=home + '/Desktop/protocols/data/')
    parser.add_argument('--embeddings',
                        type=str,
                        default=home + '/Downloads/embeddings/')
    parser.add_argument('--regressiontype',
                        type=str,
                        default="multinomial")
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--batchsize',
                        type=int,
                        default=128)
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

    # parse arguments
    args = parser.parse_args()

    # Dictionary storing the configuration of the model
    model_type = {'arg': {'repr': args.argrep, 'context': args.argcontext},
                  'pred': {'repr': args.predrep, 'context': args.predcontext}}

    arg_datafile = args.datapath + "arg_long_data.tsv"
    arg_attributes = ["part", "kind", "abs"]
    arg_attr_map = {"part": "Is.Particular", "kind": "Is.Kind", "abs": "Is.Abstract"}
    arg_attr_conf = {"part": "Part.Confidence", "kind": "Kind.Confidence",
                     "abs": "Abs.Confidence"}

    pred_datafile = args.datapath + "pred_long_data.tsv"
    pred_attributes = ["part", "hyp", "dyn"]
    pred_attr_map = {"part": "Is.Particular", "dyn": "Is.Dynamic", "hyp": "Is.Hypothetical"}
    pred_attr_conf = {"part": "Part.Confidence", "dyn": "Dyn.Confidence",
                      "hyp": "Hyp.Confidence"}

    attributes = {'arg': ['part', 'kind', 'abs'], 'pred': ['part', 'dyn', 'hyp']}
    # Load the structures/sentences
    # structures = {}
    # with open(home + '/Desktop/protocols/data/structures.tsv', 'r') as f:
    #     for line in f.readlines():
    #         structs = line.split('\t')
    #         structures[structs[0]] = structs[1].split()

    # arg_stuff = read_data(datafile=arg_datafile,
    #                       attributes=arg_attributes,
    #                       attr_map=arg_attr_map, attr_conf=arg_attr_conf,
    #                       regressiontype=args.regressiontype,
    #                       structures=structures, batch_size=args.batchsize)
    # pred_stuff = read_data(datafile=pred_datafile,
    #                        attributes=pred_attributes, attr_map=pred_attr_map,
    #                        attr_conf=pred_attr_conf,
    #                        regressiontype=args.regressiontype,
    #                        structures=structures, batch_size=args.batchsize)

    # train_data = []
    # dev_data = {'arg': None, 'pred': None}
    # dev_data['arg'] = arg_stuff[1]
    # dev_data['pred'] = pred_stuff[1]
    # for ij in range(len(arg_stuff[0])):
    #     train_data.append(interleave_lists(arg_stuff[0][ij], pred_stuff[0][ij]))
    # with open("train_stuff.pkl", "wb") as arg_out, open("dev_stuff.pkl", "wb") as pred_out:
    #     pickle.dump(train_data, arg_out)
    #     pickle.dump(dev_data, pred_out)

    train_in = open("train_stuff.pkl", "rb")
    dev_in = open("dev_stuff.pkl", "rb")
    train_data = pickle.load(train_in)
    dev_data = pickle.load(dev_in)
    train_in.close()
    dev_in.close()

    # ELMO parameters
    options_file = args.embeddings + "options/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
    weight_file = args.embeddings + "weights/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
    elmo_params = (options_file, weight_file)

    # pyTorch figures out device to do computation on
    device_to_use = device("cuda:0" if is_available() else "cpu")

    # Initialise the model
    trainer = MLPTrainer(embed_params=elmo_params, all_attrs=attributes,
                         device=device_to_use, attention_type=model_type)

    # Now to prepare all the damn inputs

    x, y, tokens, spans, context_roots, context_spans, loss_wts = train_data
    # Training phase
    trainer.fit(X=x, Y=y, loss_wts=loss_wts, tokens=tokens, spans=spans,
                context_roots=context_roots, context_spans=context_spans,
                dev=dev_data, epochs=args.epochs)

    # Save the model
