class Teller():
    def __init__(self, cfg):
        super(Teller, self).__init__()

        ########################We will need Three Components########
        self.cfg = cfg
        
        #TODO: Define Model Structure

        #TODO: Define Loss Criterion

        #TODO: Define Optimizer

    def train_batch(self, batch, epoch, iteration, visualizer, logger, total_iters=0, current_batch_t=0):
        r"""
        compute the loss and backpropagate from the loss
        """
        
        #TODO: Compute the loss and backpropagate

        #TODO: visualize the loss
    def _plot_teller_losses(self, visualizer, teller_loss, iteration):
        visualizer.plot('Teller Decoder Loss', 'train', iteration, teller_loss)

    def save_model(self, path, epoch, iteration):
        pass
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # snapshot = {
        #     'epoch': epoch,
        #     'iteration': iteration,
        #     'img_encoder_state_dict': self.img_encoder.state_dict(),
        #     'teller_dialog_encoder_state_dict': self.dialog_encoder.state_dict(),
        #     'utterance_decoder_state_dict': self.utterance_decoder.state_dict(),
        #     'teller_optimizer_state_dict': self.optimizer.state_dict(),
        #     'cfg': self.cfg,
        # }

        # torch.save(snapshot, '{}/snapshot_{}.pth'.format(path, iteration))

    def load_model(self, snapshot_path):
        pass