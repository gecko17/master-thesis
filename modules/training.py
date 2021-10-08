import torch
import time

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rouge import Rouge
from datetime import datetime, timezone

from .datasets import Seq2SeqDataset, SentenceClusterDataset


class Seq2SeqTrainer():
    def __init__(self, model, optimizer, loss_function, batch_size: int = 8, epochs: int = 3,
                 checkpoint_path: str = None, tensorboard_path: str = None, tokenizer=None, fp16: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.rouge = Rouge()
        self.batch_size = batch_size
        self.val_batch_size = 32
        self.epochs = epochs
        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        # self.model.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.scaler = torch.cuda.amp.GradScaler()
        self.tokenizer = tokenizer
        self.fp16 = fp16

        if tensorboard_path:
            log_dir = tensorboard_path + '/' + \
                datetime.utcnow().replace(tzinfo=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = SummaryWriter()

    def train_step(self, x_enc_batch, x_dec_batch, y_target_batch):
        self.optimizer.zero_grad()

        raw_scores = self.model(x_enc_batch, x_dec_batch)
        raw_scores = raw_scores.view(-1, raw_scores.shape[-1])
        y_target = y_target_batch.view(-1)

        if self.fp16:
            with torch.cuda.amp.autocast():
                loss = self.loss_function(raw_scores, y_target)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.loss_function(raw_scores, y_target)
            loss.backward()
            self.optimizer.step()

        return loss

    def run_epoch(self, train_dl: DataLoader, epoch: int):
        self.model.train()
        epoch_loss = 0
        with tqdm(train_dl, unit='steps') as tprogress:
            tprogress.set_description(f"Epoch {epoch}")
            for i, (x_enc, x_dec, targets) in enumerate(tprogress):

                loss = self.train_step(x_enc, x_dec, targets).item()
                epoch_loss += loss

                tprogress.set_postfix(loss=epoch_loss / (i + 1))

            return epoch_loss / len(train_dl)

    def run_eval(self, dl: DataLoader, generation_size: int = 4):
        self.model.eval()
        val_loss = 0
        recon_tokens = list()
        reconstructed = list()
        original_texts = list()
        generation_count = 0

        with torch.no_grad():

            with tqdm(dl, unit='steps') as tprogress:
                for i, (x_enc, x_dec, targets) in enumerate(tprogress):
                    raw_scores = self.model(x_enc, x_dec)
                    raw_scores = raw_scores.view(-1, raw_scores.shape[-1])
                    y_target = targets.view(-1)
                    loss = self.loss_function(raw_scores, y_target).item()
                    val_loss += loss

                    tprogress.set_postfix(val_loss=val_loss / (i + 1))

                    # validate reconstructed text with rouge
                    if generation_count < generation_size:
                        generation_count += 1
                        for x in x_enc['input_ids']:
                            xn = {
                                'input_ids': x.view(1, x.shape[0])
                            }

                            tokens = self.model.reconstruct(
                                xn, n_beams=3, return_all_candidates=False)[0]  # batch_size 1
                            if tokens.shape[1] < 128:
                                tokens = torch.cat(
                                    (tokens,
                                     torch.full(
                                         (1,
                                          128 -
                                          tokens.shape[1]),
                                         0).to(
                                         self.device)),
                                    dim=-
                                    1)
                            recon_tokens.append(tokens)

                        original_texts.extend(
                            self.tokenizer.batch_decode(
                                targets, skip_special_tokens=True))

            # Needs to be changed when beam allows batch input
            recon_tokens = torch.cat(recon_tokens, dim=0)
            reconstructed = self.tokenizer.batch_decode(
                recon_tokens, skip_special_tokens=True)

        #original_texts = dl.dataset.sequences
        try:
            scores = self.rouge.get_scores(
                original_texts, reconstructed, avg=True)
        except ValueError as ve:
            scores = None  # rouge is ignoring full stops, hence can lead to empty sequences

        return val_loss / len(dl), scores

    def train(self, data: Seq2SeqDataset, val_size: int = 0.1,
              eval: bool = False, generation_size: int = 4):
        start_time = time.time()

        test_size = int(len(data) * val_size)
        train_size = len(data) - test_size
        train, test = random_split(data, [train_size, test_size])
        train_dl = DataLoader(train, batch_size=self.batch_size)
        test_dl = DataLoader(test, batch_size=self.batch_size)

        scores = None

        #self.writer.add_text("Config/Encoder", self.model.encoder.config.to_json_str())
        #self.writer.add_text("Config/Decoder", self.model.decoder.config.to_json_str())

        for epoch in range(self.epochs):
            epoch_time = time.time()
            print(f'Running Epoch {epoch+1} ...')

            loss = self.run_epoch(train_dl, epoch + 1)
            print('', end='\n')
            print(f'Epoch Loss: {loss}')
            print(f'Epoch Time Elapsed: {time.time() - epoch_time}')
            self.writer.add_scalar("Loss/train", loss, epoch)

            if eval:
                eval_loss, scores = self.run_eval(test_dl, generation_size)
                print('', end='\n')
                print(f'Validation Loss: {eval_loss}')
                print(f'Rouge Scores: {scores}')
                self.writer.add_scalar("Loss/test", eval_loss, epoch)
                self.writer.add_scalars(
                    "Loss", {"train": loss, "test": eval_loss}, epoch)

            if scores:
                self.writer.add_scalars(
                    'Scores/rouge-1', scores['rouge-1'], epoch)
                self.writer.add_scalars(
                    'Scores/rouge-2', scores['rouge-2'], epoch)
                self.writer.add_scalars(
                    'Scores/rouge-l', scores['rouge-l'], epoch)

            self.writer.flush()

            if self.checkpoint_path:
                path = self.checkpoint_path + \
                    '/checkpoint-epoch-' + str(epoch + 1)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'eval_loss': eval_loss,
                    'encoder_config': self.model.encoder.config.to_json_dict(),
                    'decoder_config': self.model.decoder.config.to_json_dict()
                }, path)

        self.writer.add_hparams({
            'lr': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.batch_size,
            'n_train_samples': train_size,
            'n_epochs': self.epochs,
            'encoder_config': self.model.encoder.config.to_json_str(),
            'decoder_config': self.model.decoder.config.to_json_str()
        }, {
            'final_loss': loss,
            'final_test_loss:': eval_loss,
            'final_r2_f': scores['rouge-2']['f']
        })
        self.writer.flush()
        self.writer.close()

        print(f'Total Time Elapsed: {time.time() - start_time}')


class SummaryTrainer():
    def __init__(self, model, optimizer, loss_functions: dict, epochs: int = 3,
                 checkpoint_path: str = None, tensorboard_path: str = None, tokenizer=None, fp16: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.loss_functions = loss_functions
        self.rouge = Rouge()
        self.epochs = epochs
        try:
            self.device = xm.xla_device()
        except BaseException:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        print(f'Using device: {self.device}')
        # self.model.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.scaler = torch.cuda.amp.GradScaler()
        self.tokenizer = tokenizer
        self.fp16 = fp16

        if len(self.loss_functions.keys()) > 1:
            if 'reconstruction_loss' in self.loss_functions and 'similarity_loss' in self.loss_functions:
                self.reconstruction_only = False
            else:
                raise ValueError("Wrong loss functions provided")
        else:
            # only reconstruction loss
            if 'reconstruction_loss' in self.loss_functions:
                self.reconstruction_only = True
            else:
                raise ValueError("Wrong loss functions provided")

        if tensorboard_path:
            log_dir = tensorboard_path + '/' + \
                datetime.utcnow().replace(tzinfo=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = SummaryWriter()

    def train_step(self, X_enc, X_dec, Y):
        reconstruction_loss_function = self.loss_functions['reconstruction_loss']
        if not self.reconstruction_only:
            similarity_loss_function = self.loss_functions['similarity_loss']

        self.optimizer.zero_grad()

        # step 1: reconstruction loss
        raw_scores = self.model(X_enc, X_dec)
        raw_scores = raw_scores.view(-1, raw_scores.shape[-1])
        y = Y.view(-1)

        if self.fp16:
            with torch.cuda.amp.autocast():
                reconstruction_loss = reconstruction_loss_function(
                    raw_scores, y)

        else:
            reconstruction_loss = reconstruction_loss_function(raw_scores, y)

        if not self.reconstruction_only:
            # step 2: similarity loss
            _, h_n, c_n = self.model.encoder(X_enc)
            h_0 = torch.mean(h_n, dim=1, keepdim=True)
            c_0 = torch.mean(c_n, dim=1, keepdim=True)

            summary, _ = self.model.generate_from_encoding(h_0, c_0)
            _, back_encoded_h, back_encoded_c = self.model.encoder(
                {'input_ids': summary})

            similarity_loss = 0
            batch_size = h_n.shape[1]

            for i in range(batch_size):
                h_i = h_n[:, i, :]
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        similarity_loss += (1 -
                                            similarity_loss_function(h_i, back_encoded_h))
                else:
                    similarity_loss += (1 -
                                        similarity_loss_function(h_i, back_encoded_h))

            similarity_loss = similarity_loss / batch_size

            # step 3: combine losses
            loss = reconstruction_loss + similarity_loss

        else:
            loss = reconstruction_loss

        if self.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
            # xm.optimizer_step(optimizer)
            # xm.mark_step()

        return loss

    def run_epoch(self, train_dl: DataLoader, epoch: int):
        self.model.train()
        epoch_loss = 0
        with tqdm(train_dl, unit='steps') as tprogress:
            tprogress.set_description(f"Epoch {epoch}")
            for i, (x_enc, x_dec, targets) in enumerate(tprogress):

                loss = self.train_step(x_enc, x_dec, targets).item()
                epoch_loss += loss

                tprogress.set_postfix(loss=epoch_loss / (i + 1))

            return epoch_loss / len(train_dl)

    def run_eval(self, dl: DataLoader, generation_size: int = 4):
        self.model.eval()
        val_loss = 0
        recon_tokens = list()
        reconstructed = list()
        original_texts = list()
        sum_original_texts = list()
        sum_summaries = list()
        generation_count = 0
        reconstruction_loss_function = self.loss_functions['reconstruction_loss']
        if not self.reconstruction_only:
            similarity_loss_function = self.loss_functions['similarity_loss']

        with torch.no_grad():

            with tqdm(dl, unit='steps') as tprogress:
                for i, (x_enc, x_dec, targets) in enumerate(tprogress):
                    raw_scores = self.model(x_enc, x_dec)
                    raw_scores = raw_scores.view(-1, raw_scores.shape[-1])
                    y_target = targets.view(-1)
                    reconstruction_loss = reconstruction_loss_function(
                        raw_scores, y_target).item()
                    #val_loss += loss

                    if not self.reconstruction_only:
                        _, h_n, c_n = self.model.encoder(x_enc)
                        h_0 = torch.mean(h_n, dim=1, keepdim=True)
                        c_0 = torch.mean(c_n, dim=1, keepdim=True)

                        summary, _ = self.model.generate_from_encoding(
                            h_0, c_0)
                        _, back_encoded_h, back_encoded_c = self.model.encoder(
                            {'input_ids': summary})

                        similarity_loss = 0
                        batch_size = h_n.shape[1]

                        for j in range(batch_size):
                            h_i = h_n[:, j, :]
                            if self.fp16:
                                with torch.cuda.amp.autocast():
                                    similarity_loss += (1 -
                                                        similarity_loss_function(h_i, back_encoded_h).item())
                            else:
                                similarity_loss += (1 -
                                                    similarity_loss_function(h_i, back_encoded_h).item())

                        similarity_loss = similarity_loss / batch_size

                        # step 3: combine losses
                        loss = reconstruction_loss + similarity_loss

                        # validate summary rouge with each sequence
                        sum_original_texts.extend(
                            self.tokenizer.batch_decode(
                                targets, skip_special_tokens=True))
                        sum_summaries.extend(
                            self.tokenizer.batch_decode(
                                summary.expand(
                                    targets.shape[0], -1), skip_special_tokens=True))

                    else:
                        loss = reconstruction_loss

                    val_loss += loss

                    tprogress.set_postfix(val_loss=val_loss / (i + 1))

                    # validate reconstructed text with rouge
                    if generation_count < generation_size:
                        generation_count += 1
                        for x in x_enc['input_ids']:
                            xn = {
                                'input_ids': x.view(1, x.shape[0])
                            }

                            tokens = self.model.reconstruct(
                                xn, n_beams=3, return_all_candidates=False)[0]  # batch_size 1
                            if tokens.shape[1] < 128:
                                tokens = torch.cat(
                                    (tokens,
                                     torch.full(
                                         (1,
                                          128 -
                                          tokens.shape[1]),
                                         0).to(
                                         self.device)),
                                    dim=-
                                    1)
                            recon_tokens.append(tokens)

                        original_texts.extend(
                            self.tokenizer.batch_decode(
                                targets, skip_special_tokens=True))

                        if self.reconstruction_only:
                            sum_original_texts.extend(self.tokenizer.batch_decode(
                                targets, skip_special_tokens=True))

                            _, h_n, c_n = self.model.encoder(x_enc)
                            h_0 = torch.mean(h_n, dim=1, keepdim=True)
                            c_0 = torch.mean(c_n, dim=1, keepdim=True)
                            summary, _ = self.model.generate_from_encoding(
                                h_0, c_0)

                            sum_summaries.extend(
                                self.tokenizer.batch_decode(
                                    summary.expand(
                                        targets.shape[0], -1), skip_special_tokens=True))

            # Needs to be changed when beam allows batch input
            recon_tokens = torch.cat(recon_tokens, dim=0)
            reconstructed = self.tokenizer.batch_decode(
                recon_tokens, skip_special_tokens=True)

        #original_texts = dl.dataset.sequences
        try:
            scores = self.rouge.get_scores(
                reconstructed, original_texts, avg=True)
            summary_scores = self.rouge.get_scores(
                sum_summaries, sum_original_texts, avg=True)
        except ValueError as ve:
            scores = None  # rouge is ignoring full stops, hence can lead to empty sequences
            summary_scores = None

        return val_loss / len(dl), scores, summary_scores

    def train(self, train_data: torch.utils.data.Dataset, test_data: torch.utils.data.Dataset, val_size: int = 0.1,
              eval: bool = False, generation_size: int = 4, from_epoch: int = 0):
        start_time = time.time()

        #test_size = int(len(data) * val_size)
        #train_size = len(data) - test_size
        #train, test = random_split(data, [train_size, test_size])
        train_dl = DataLoader(train_data, batch_size=None)
        test_dl = DataLoader(test_data, batch_size=None)
        train_size = len(train_data)

        scores = None
        summary_scores = None
        eval_loss = None

        #self.writer.add_text("Config/Encoder", self.model.encoder.config.to_json_str())
        #self.writer.add_text("Config/Decoder", self.model.decoder.config.to_json_str())

        for epoch in range(self.epochs):
            epoch_time = time.time()
            epoch += from_epoch
            print(f'Running Epoch {epoch+1} ...')

            loss = self.run_epoch(train_dl, epoch + 1)
            print('', end='\n')
            print(f'Epoch Loss: {loss}')
            print(f'Epoch Time Elapsed: {time.time() - epoch_time}')
            self.writer.add_scalar("Loss/train", loss, epoch)

            if eval:
                eval_loss, scores, summary_scores = self.run_eval(
                    test_dl, generation_size)
                print('', end='\n')
                print(f'Validation Loss: {eval_loss}')
                print(f'Reconstruction Rouge Scores: {scores}')
                print(f'Summary Rouge Scores: {summary_scores}')
                self.writer.add_scalar("Loss/test", eval_loss, epoch)
                self.writer.add_scalars(
                    "Loss", {"train": loss, "test": eval_loss}, epoch)

            if scores:
                self.writer.add_scalars(
                    'Reconstruction-Scores/rouge-1', scores['rouge-1'], epoch)
                self.writer.add_scalars(
                    'Reconstruction-Scores/rouge-2', scores['rouge-2'], epoch)
                self.writer.add_scalars(
                    'Reconstruction-Scores/rouge-l', scores['rouge-l'], epoch)

            if summary_scores:
                self.writer.add_scalars(
                    'Summary-Scores/rouge-1', summary_scores['rouge-1'], epoch)
                self.writer.add_scalars(
                    'Summary-Scores/rouge-2', summary_scores['rouge-2'], epoch)
                self.writer.add_scalars(
                    'Summary-Scores/rouge-l', summary_scores['rouge-l'], epoch)

            self.writer.flush()

            if self.checkpoint_path:
                path = self.checkpoint_path + \
                    '/checkpoint-epoch-' + str(epoch + 1)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'eval_loss': eval_loss,
                    'encoder_config': self.model.encoder.config.to_json_dict(),
                    'decoder_config': self.model.decoder.config.to_json_dict()
                }, path)

        self.writer.add_hparams({
            'lr': self.optimizer.param_groups[0]['lr'],
            # 'batch_size': self.batch_size,
            'n_train_samples': train_size,
            'n_epochs': self.epochs,
            'encoder_config': self.model.encoder.config.to_json_str(),
            'decoder_config': self.model.decoder.config.to_json_str()
        }, {
            'final_loss': loss,
            'final_test_loss:': eval_loss if eval_loss else 0,
            'final_r2_f': scores['rouge-2']['f'] if scores else 0
        })
        self.writer.flush()
        self.writer.close()

        print(f'Total Time Elapsed: {time.time() - start_time}')
