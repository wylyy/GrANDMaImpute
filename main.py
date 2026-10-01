
import torch

from models import twins_process,AE_process
import random
import numpy as np
from tqdm import tqdm
from loss import TwinsLoss
from loss import SCEMSELoss
import pandas as pd
from utils import create_optimizer,build_args
import logging
import matplotlib.pyplot as plt

from data_enhancement import drop_nodes,knn_graph


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
random_seed =4

random.seed(random_seed)  # set random seed for python
np.random.seed(random_seed )
torch.manual_seed(random_seed )
torch.cuda.manual_seed_all(random_seed )
torch.backends.cudnn.deterministic = True

def pre_main_twins(train_dataset,val_dataset,args):
    device = "cpu"
    max_epoch=args.max_epoch_f
    use_scheduler=args.scheduler

    graph_train_1, num_features = drop_nodes(train_dataset, 0.5)
    graph_train_2, num_features= drop_nodes(train_dataset, 0.5)

    graph_val_1, num_features= drop_nodes(val_dataset, 0.5)
    graph_val_2, num_features= drop_nodes(val_dataset, 0.5)

    args.num_features = num_features

    encoder1 = twins_process(args)
    encoder2 = twins_process(args)

    epoch_iter = tqdm(range(max_epoch))
    loss_fc=TwinsLoss(device)

    lr=args.lr_f
    weight_decay=args.weight_decay_f
    optimizer = torch.optim.Adam(list(encoder1.parameters()) + list(encoder2.parameters()), lr=lr,weight_decay=weight_decay)

    if use_scheduler:
        logging.info("Use schedular")

        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= scheduler)

    else:
        scheduler = None

    train_loss_z=[]
    val_loss_z=[]


    for epoch in epoch_iter:

        encoder1.train()
        encoder2.train()

        encoded_train1 = encoder1(graph_train_1,graph_train_1.ndata["feat"])
        encoded_train2 = encoder2(graph_train_2,graph_train_2.ndata["feat"])

        train_loss=loss_fc(encoded_train1,encoded_train2)


        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()


        train_loss_1 = train_loss.item()

        train_loss_z.append(train_loss_1)

        print("epoch_" + str(epoch) + "    " + "train_loss_" + str(train_loss))


        with torch.no_grad():
            encoder1.eval()
            encoder2.eval()
            encoded_val1 = encoder1(graph_val_1, graph_val_1.ndata["feat"])
            encoded_val2 = encoder2(graph_val_2, graph_val_2.ndata["feat"])

            val_loss = loss_fc(encoded_val1, encoded_val2)

            val_loss_1 = val_loss.item()

            val_loss_z.append(val_loss_1)

            print("epoch_"+str(epoch)+"    "+"val_loss_"+str(val_loss))

            if earlyStop(val_loss_z, epoch, max_epoch):
                torch.save(encoder2.state_dict(), 'shared_encoder.pth')

                break

def earlyStop(loss_list, epoch,max_epoch):

    if (epoch == max_epoch - 1):
        return True
    elif (epoch > 50) and (loss_list[epoch] > loss_list[epoch - 1]) and (
            loss_list[epoch - 1] > loss_list[epoch - 2])and (
            loss_list[epoch - 2] > loss_list[epoch - 3])and (
            loss_list[epoch - 3] > loss_list[epoch - 4])  :
        return True
    elif (epoch > 50) and (loss_list[epoch] == loss_list[epoch - 1]) and (
            loss_list[epoch - 1] == loss_list[epoch - 2])and (
            loss_list[epoch - 2] == loss_list[epoch - 3]) and (
            loss_list[epoch - 3] == loss_list[epoch - 4]):
        return True
    else:
        return False


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not np.array_equal(p1.cpu().numpy(), p2.cpu().numpy()):
            return False
    return True

def model_train(train_dataset,val_dataset,args):
    device = "cpu"

    max_epoch = args.max_epoch

    optim_type = args.optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    use_scheduler = args.scheduler
    gamma=args.gamma

    graph_train, num_features = knn_graph(train_dataset)

    graph_val, num_features= knn_graph(val_dataset)

    args.num_features = num_features

    ae_model =AE_process(args)

    encoder_model =twins_process(args)

    encoder_model.load_state_dict(torch.load('shared_encoder.pth'))

    ae_model.ae.encoder = encoder_model.ae.encoder

    epoch_iter = tqdm(range(max_epoch))

    optimizer = create_optimizer(optim_type, ae_model, lr, weight_decay)


    if use_scheduler:
        logging.info("Use schedular")
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    else:
        scheduler = None

    ae_model.train()
    train_loss_z = []
    val_loss_z=[]

    loss_rc = SCEMSELoss(device)

    for epoch in epoch_iter:

        ae_model.train()

        x_bar= ae_model(graph_train, graph_train.ndata["feat"])
        x_bar = x_bar.to(torch.float64)

        mse_loss=loss_rc.cal(x_bar, graph_train, gamma )

        loss = mse_loss


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        train_loss = loss.item()

        train_loss_z.append(mse_loss.item())

        print("epoch_"+str(epoch)+"    "+"train_loss_"+str(train_loss ))
        print("epoch_" + str(epoch) + "    " + "train_mse_loss_" + str(mse_loss))


        with torch.no_grad():
            ae_model.eval()
            x_val,x_init,x_rec = ae_model(graph_val, graph_val.ndata["feat"])
            x_val = x_val.to(torch.float64)

            val_mse_loss= loss_rc.cal(x_val,graph_val,x_init,x_rec, gamma )

            val_loss =val_mse_loss

            val_loss_z.append(val_mse_loss.item())

            print("epoch_" + str(epoch) + "    " + "val_loss_" + str(val_loss))
            print("epoch_" + str(epoch) + "    " + "val_mse_loss_" + str(val_mse_loss))


            if earlyStop(val_loss_z, epoch, max_epoch):
                return ae_model

                break

