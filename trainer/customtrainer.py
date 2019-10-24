import torch
from torch.nn import functional as F
from models.Flowestimator import FlowEstimator
import pytorch_lightning as pl
from utils.warper import warper
from dataloader.sintelloader import SintelLoader
from utils.photometricloss import photometricloss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.averagemeter import AverageMeter
from torchvision.utils import make_grid
from utils.flow2rgb import flow2rgb
from utils.replicateto3channel import replicatechannel



class FlowTrainer(object):
    def __init__(self):
        super(FlowTrainer, self).__init__()
        # not the best model...
        self.model = FlowEstimator(shape=(256, 256),
                                   use_l2=True,
                                   channel_in=3,
                                   stride=4,
                                   kernel_size=7,
                                   use_cst=False)
        self.optimizer = None
        self.lr_scheduler = None
        self.save_dir = None

        self.epoch = 1000

        self.train_loader = SintelLoader(batch_size=20,
                                         pin_memory=True,
                                         num_workers=8,)

        self.val_loader = None

        self.test_loader = SintelLoader(sintel_root="/data/keshav/sintel/test/final",
                                        batch_size=1,
                                        pin_memory=True,
                                        num_workers=8)

        self.sample_test = [*SintelLoader(sintel_root="/data/keshav/sintel/test/final",
                                          test=True, nsample=10, visualize=True).load()][0]
        self.sample_train = [*SintelLoader(nsample=10,visualize=True).load()][0]

        self.sample_val = None

        self.save_model_path = './best/'
        self.load_model_path = None
        self.best_metrics = {'train_loss': None,
                             'val_loss': None}
        self.gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]

        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001)
        # self.scheduler = ReduceLROnPlateau(self.optimizer)
        self.scheduler = CosineAnnealingLR(self.optimizer, len(self.train_loader.load()))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.photoloss = torch.nn.MSELoss()

        self.writer = SummaryWriter()
        self.global_step = 0

    def resetsample(self):
        self.sample_test = [*SintelLoader(sintel_root="/data/keshav/sintel/test/final",
                                          test=True, nsample=10, visualize=True).load()][0]
        self.sample_train = [*SintelLoader(nsample=10, visualize=True).load()][0]

    def initialize(self):
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
        if self.load_model_path:
            # LOAD MODEL WEIGHTS HERE
            pass
        self.initialized = True

    def savemodel(self, metrics, compare='val_loss'):
        # Save model in save_model_path
        if self.best_metrics.get('val_loss') > metrics.get('val_loss'):
            # save only if new metrics are low
            self.best_metrics.update(metrics)
            pass
        else:
            # Load from the best saved
            pass

    def train_epoch_end(self, metrics):
        self.resetsample()
        self.model.eval()
        with torch.no_grad():
            frame1 = self.sample_train['frame1'].to(self.device)
            frame2 = self.sample_train['frame2'].to(self.device)

            frame1Unet = self.sample_train['frame1Unet'].to(self.device)
            frame2Unet = self.sample_train['frame2Unet'].to(self.device)


            flow, occ = self.model(frame1, frame2)
            frame2_ = warper(flow, frame1)

            occ = replicatechannel(occ)

            #without unet
            # sampocc = replicatechannel(self.sample_train['occlusion'].cuda())
            #with unet
            sampocc = replicatechannel(self.sample_train['occlusionUnet'].cuda())

            occs = torch.cat([sampocc, occ])
            occs = make_grid(occs, nrow=10).unsqueeze(0)

            #without unet
            # frames = torch.cat([frame1_, frame1, frame2])
            # with unet
            frames = torch.cat([frame2_, frame2Unet, frame1Unet])

            frames = make_grid(frames, nrow=10).unsqueeze(0)

            #without unet
            # flows = torch.cat([flow2rgb(flow.cpu()).cuda(), self.sample_train['flow'].cuda()])
            # with unet
            flows = torch.cat([flow2rgb(flow.cpu()).cuda(), self.sample_train['flowUnet'].cuda()])
            flows = make_grid(flows, nrow=10).unsqueeze(0)

            self.writer.add_images('TRAIN/Frames', frames, metrics.get('nb_batch'))
            self.writer.add_images('TRAIN/Flows', flows, metrics.get('nb_batch'))
            self.writer.add_images('TRAIN/Occlusions', occs, metrics.get('nb_batch'))

        return self.val(metrics)

    def val_end(self, metrics):
        return metrics

    def test_end(self, metrics):
        with torch.no_grad():
            frame1 = self.sample_test['frame1'].to(self.device)
            frame2 = self.sample_test['frame2'].to(self.device)

            frame1Unet = self.sample_test['frame1Unet'].to(self.device)
            frame2Unet = self.sample_test['frame2Unet'].to(self.device)

            flow, occ = self.model(frame1, frame2)
            frame1_ = warper(flow, frame2)
            occ = replicatechannel(occ)

            frames = torch.cat([frame1_, frame1Unet, frame2Unet, flow2rgb(flow.cpu()).cuda(), occ])
            frames = make_grid(frames, nrow=10).unsqueeze(0)

            self.writer.add_images('TEST/Frames', frames, metrics.get('nb_batch'))
        return metrics

    def train(self, nb_epoch):
        trainstream = tqdm(self.train_loader.load())
        self.avg_loss = AverageMeter()
        self.model.train()
        for i, data in enumerate(trainstream):
            self.global_step += 1
            trainstream.set_description('TRAINING')

            # GET X and Frame 2
            # wdt = data['displacement'].to(self.device)
            frame2 = data['frame2'].to(self.device)
            frame1 = data['frame1'].to(self.device)

            frame1Unet = data['frame1Unet'].to(self.device)
            frame2Unet = data['frame2Unet'].to(self.device)
            self.optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                flow, occ = self.model(frame1, frame2)
                frame1_ = warper(flow, frame2)
                #WITHOUT UNET
                # loss = photometricloss(frame1, frame1_, occ)
                #WITH UNET
                loss = photometricloss(frame1Unet, frame1_,frame2Unet, occ)
                self.avg_loss.update(loss.item(), i + 1)
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar('Loss/train',
                                       self.avg_loss.avg, self.global_step)

                trainstream.set_postfix({'epoch': nb_epoch,
                                         'loss': self.avg_loss.avg})
        self.scheduler.step(loss)
        trainstream.close()
        return self.train_epoch_end({'TRloss': self.avg_loss.avg, 'epoch': nb_epoch, })

    def val(self, metrics):
        if self.val_loader is None: return self.test(metrics)
        # DO VAL STUFF HERE
        valstream = tqdm(self.val_loader.load())
        for data in valstream:
            pass
        return self.val_end(metrics)

    def test(self, metrics={}):
        teststream = tqdm(self.test_loader.load())
        self.avg_loss = AverageMeter()
        with torch.no_grad():
            for i, data in enumerate(teststream):
                teststream.set_description('TESTING')
                frame2 = data['frame2'].to(self.device)
                frame1 = data['frame1'].to(self.device)

                frame2Unet = data['frame2Unet'].to(self.device)
                frame1Unet = data['frame1Unet'].to(self.device)


                flow, occ = self.model(frame1, frame2)
                frame1_ = warper(flow, frame2)
                loss = photometricloss(frame1Unet, frame1_,frame2Unet, occ)
                self.avg_loss.update(loss.item(), i + 1)
                metrics.update({'TSloss': self.avg_loss.avg})
                teststream.set_postfix(metrics)
                self.writer.add_scalar('Loss/test', self.avg_loss.avg, i)
        teststream.close()

        return self.test_end(metrics)

    def loggings(self, **metrics):
        pass

    def run(self):
        self.initialize()
        for i in range(self.epoch):
            metrics = self.train(i)
        self.test(metrics)
        self.writer.close()