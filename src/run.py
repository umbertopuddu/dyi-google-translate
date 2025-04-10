from constants import *
from custom_data import *
from transformer import *
from data_structure import *
import torch
import argparse
import datetime
import copy
import sentencepiece as spm
import sys
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

class Manager():
    def __init__(self, is_train=True, ckpt_name=None):
        # Load vocabs
        print("Loading vocabs...")
        self.src_i2w = {}
        self.trg_i2w = {}

        with open(f"{SP_DIR}/{src_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.src_i2w[i] = word

        with open(f"{SP_DIR}/{trg_model_prefix}.vocab") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            word = line.strip().split('\t')[0]
            self.trg_i2w[i] = word

        print(f"The size of src vocab is {len(self.src_i2w)} and that of trg vocab is {len(self.trg_i2w)}.")

        # Load Transformer model & Adam optimizer
        print("Loading Transformer model & Adam optimizer...")
        self.model = Transformer(src_vocab_size=len(self.src_i2w), trg_vocab_size=len(self.trg_i2w)).to(device)
        
        # If more than one GPU is available, wrap the model with DataParallel for parallel training
        if num_gpus > 1:
            print(f"Multiple GPUs detected ({num_gpus}). Using DataParallel for parallelization.")
            self.model = nn.DataParallel(self.model)
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_loss = sys.float_info.max

        if ckpt_name is not None:
            assert os.path.exists(f"{ckpt_dir}/{ckpt_name}"), f"There is no checkpoint named {ckpt_name}."

            print("Loading checkpoint...")
            if(num_gpus == 0):
                checkpoint = torch.load(f"{ckpt_dir}/{ckpt_name}", map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(f"{ckpt_dir}/{ckpt_name}", weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.best_loss = checkpoint['loss']
        else:
            print("Initializing the model...")
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if is_train:
            # Load loss function
            print("Loading loss function...")
            self.criterion = nn.NLLLoss()

            # Load dataloaders
            print("Loading dataloaders...")
            self.train_loader = get_data_loader(TRAIN_NAME)
            self.valid_loader = get_data_loader(VALID_NAME)

        print("Setting finished.")

    def train(self):
        print("Training starts.")

        for epoch in range(1, num_epochs+1):
            self.model.train()
            
            train_losses = []
            start_time = datetime.datetime.now()
            
            # Determine logging interval (quarter of an epoch)
            total_batches = len(self.train_loader)
            quarter_epoch_interval = max(1, total_batches // 4)
            
            for i, batch in tqdm(enumerate(self.train_loader), total=total_batches):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask)  # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                self.optim.zero_grad()
                loss = self.criterion(
                    output.view(-1, sp_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                loss.backward()
                self.optim.step()

                train_losses.append(loss.item())
                
                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()
                
                # Log metrics every quarter of the epoch
                if (i+1) % quarter_epoch_interval == 0 or (i+1) == total_batches:
                    elapsed = datetime.datetime.now() - start_time
                    avg_loss_so_far = np.mean(train_losses)
                    progress_pct = ((i+1) / total_batches) * 100
                    print(f"Epoch {epoch}: {progress_pct:.0f}% done - Avg Loss so far: {avg_loss_so_far:.4f}, Elapsed Time: {elapsed}")

            end_time = datetime.datetime.now()
            training_time = end_time - start_time
            seconds = training_time.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60

            mean_train_loss = np.mean(train_losses)
            print(f"#################### Epoch: {epoch} ####################")
            print(f"Train loss: {mean_train_loss} || One epoch training time: {hours}hrs {minutes}mins {seconds}secs")

            valid_loss, valid_time = self.validation()
            
            if valid_loss < self.best_loss:
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                    
                self.best_loss = valid_loss
                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'loss': self.best_loss
                }
                torch.save(state_dict, f"{ckpt_dir}/best_ckpt.tar")
                print(f"***** Current best checkpoint is saved. *****")

            print(f"Best valid loss: {self.best_loss}")
            print(f"Valid loss: {valid_loss} || One epoch training time: {valid_time}")

    print(f"Training finished!")

    def validation(self):
        print("Validation processing...")
        self.model.eval()
        
        valid_losses = []
        start_time = datetime.datetime.now()

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.valid_loader)):
                src_input, trg_input, trg_output = batch
                src_input, trg_input, trg_output = src_input.to(device), trg_input.to(device), trg_output.to(device)

                e_mask, d_mask = self.make_mask(src_input, trg_input)

                output = self.model(src_input, trg_input, e_mask, d_mask)  # (B, L, vocab_size)

                trg_output_shape = trg_output.shape
                loss = self.criterion(
                    output.view(-1, sp_vocab_size),
                    trg_output.view(trg_output_shape[0] * trg_output_shape[1])
                )

                valid_losses.append(loss.item())

                del src_input, trg_input, trg_output, e_mask, d_mask, output
                torch.cuda.empty_cache()

        end_time = datetime.datetime.now()
        validation_time = end_time - start_time
        seconds = validation_time.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        mean_valid_loss = np.mean(valid_losses)
        
        return mean_valid_loss, f"{hours}hrs {minutes}mins {seconds}secs"

    def inference(self, input_sentence, method):
        print("Inference starts.")
        self.model.eval()

        print("Loading sentencepiece tokenizer...")
        src_sp = spm.SentencePieceProcessor()
        trg_sp = spm.SentencePieceProcessor()
        src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
        trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")

        print("Preprocessing input sentence...")
        tokenized = src_sp.EncodeAsIds(input_sentence)
        src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device)  # (1, L)
        e_mask = (src_data != pad_id).unsqueeze(1).to(device)  # (1, 1, L)

        start_time = datetime.datetime.now()

        print("Encoding input sentence...")
        src_data = self.model.src_embedding(src_data)
        src_data = self.model.positional_encoder(src_data)
        e_output = self.model.encoder(src_data, e_mask)  # (1, L, d_model)

        if method == 'greedy':
            print("Greedy decoding selected.")
            result = self.greedy_search(e_output, e_mask, trg_sp)
        elif method == 'beam':
            print("Beam search selected.")
            result = self.beam_search(e_output, e_mask, trg_sp)

        end_time = datetime.datetime.now()

        total_inference_time = end_time - start_time
        seconds = total_inference_time.seconds
        minutes = seconds // 60
        seconds = seconds % 60

        print(f"Input: {input_sentence}")
        print(f"Result: {result}")
        print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
        
    def greedy_search(self, e_output, e_mask, trg_sp):
        last_words = torch.LongTensor([pad_id] * seq_len).to(device)  # (L)
        last_words[0] = sos_id  # (L)
        cur_len = 1

        for i in range(seq_len):
            d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device)  # (1, 1, L)
            nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
            nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) triangular shape
            d_mask = d_mask & nopeak_mask  # (1, L, L)

            trg_embedded = self.model.trg_embedding(last_words.unsqueeze(0))
            trg_positional_encoded = self.model.positional_encoder(trg_embedded)
            decoder_output = self.model.decoder(
                trg_positional_encoded,
                e_output,
                e_mask,
                d_mask
            )  # (1, L, d_model)

            output = self.model.softmax(
                self.model.output_linear(decoder_output)
            )  # (1, L, trg_vocab_size)

            output = torch.argmax(output, dim=-1)  # (1, L)
            last_word_id = output[0][i].item()
            
            if i < seq_len-1:
                last_words[i+1] = last_word_id
                cur_len += 1
            
            if last_word_id == eos_id:
                break

        if last_words[-1].item() == pad_id:
            decoded_output = last_words[1:cur_len].tolist()
        else:
            decoded_output = last_words[1:].tolist()
        decoded_output = trg_sp.decode_ids(decoded_output)
        
        return decoded_output
    
    def beam_search(self, e_output, e_mask, trg_sp):
        cur_queue = PriorityQueue()
        for k in range(beam_size):
            cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))
        
        finished_count = 0
        
        for pos in range(seq_len):
            new_queue = PriorityQueue()
            for k in range(beam_size):
                node = cur_queue.get()
                if node.is_finished:
                    new_queue.put(node)
                else:
                    trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device)  # (L)
                    d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device)  # (1, 1, L)
                    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                    nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L)
                    d_mask = d_mask & nopeak_mask  # (1, L, L)
                    
                    trg_embedded = self.model.trg_embedding(trg_input.unsqueeze(0))
                    trg_positional_encoded = self.model.positional_encoder(trg_embedded)
                    decoder_output = self.model.decoder(
                        trg_positional_encoded,
                        e_output,
                        e_mask,
                        d_mask
                    )  # (1, L, d_model)

                    output = self.model.softmax(
                        self.model.output_linear(decoder_output)
                    )  # (1, L, trg_vocab_size)
                    
                    output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                    last_word_ids = output.indices.tolist()  # (k)
                    last_word_prob = output.values.tolist()  # (k)
                    
                    for i, idx in enumerate(last_word_ids):
                        new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                        if idx == eos_id:
                            new_node.prob = new_node.prob / float(len(new_node.decoded))
                            new_node.is_finished = True
                            finished_count += 1
                        new_queue.put(new_node)
            
            cur_queue = copy.deepcopy(new_queue)
            
            if finished_count == beam_size:
                break
        
        decoded_output = cur_queue.get().decoded
        
        if decoded_output[-1] == eos_id:
            decoded_output = decoded_output[1:-1]
        else:
            decoded_output = decoded_output[1:]
            
        return trg_sp.decode_ids(decoded_output)
        

    def make_mask(self, src_input, trg_input):
        e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
        d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L)
        d_mask = d_mask & nopeak_mask  # (B, L, L)

        return e_mask, d_mask

def inference(self, input_sentence, method):
    print("Inference starts.")
    self.model.eval()
    print("Loading sentencepiece tokenizer...")
    src_sp = spm.SentencePieceProcessor()
    trg_sp = spm.SentencePieceProcessor()
    src_sp.Load(f"{SP_DIR}/{src_model_prefix}.model")
    trg_sp.Load(f"{SP_DIR}/{trg_model_prefix}.model")
    print("Preprocessing input sentence...")
    tokenized = src_sp.EncodeAsIds(input_sentence)
    src_data = torch.LongTensor(pad_or_truncate(tokenized)).unsqueeze(0).to(device)  # (1, L)
    e_mask = (src_data != pad_id).unsqueeze(1).to(device)  # (1, 1, L)
    start_time = datetime.datetime.now()
    print("Encoding input sentence...")
    src_data = self.model.src_embedding(src_data)
    src_data = self.model.positional_encoder(src_data)
    e_output = self.model.encoder(src_data, e_mask)  # (1, L, d_model)
    if method == 'greedy':
        print("Greedy decoding selected.")
        result = self.greedy_search(e_output, e_mask, trg_sp)
    elif method == 'beam':
        print("Beam search selected.")
        result = self.beam_search(e_output, e_mask, trg_sp)
    else:
        raise ValueError("Invalid decoding method. Choose 'greedy' or 'beam'.")
    end_time = datetime.datetime.now()
    total_inference_time = end_time - start_time
    seconds = total_inference_time.seconds
    minutes = seconds // 60
    seconds = seconds % 60
    print(f"Input: {input_sentence}")
    print(f"Result: {result}")
    print(f"Inference finished! || Total inference time: {minutes}mins {seconds}secs")
    
def greedy_search(self, e_output, e_mask, trg_sp):
    last_words = torch.LongTensor([pad_id] * seq_len).to(device)  # (L)
    last_words[0] = sos_id  # (L)
    cur_len = 1
    for i in range(seq_len):
        d_mask = (last_words.unsqueeze(0) != pad_id).unsqueeze(1).to(device)  # (1, 1, L)
        nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)  # (1, L, L)
        nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L) triangular shape
        d_mask = d_mask & nopeak_mask  # (1, L, L)
        trg_embedded = self.model.trg_embedding(last_words.unsqueeze(0))
        trg_positional_encoded = self.model.positional_encoder(trg_embedded)
        decoder_output = self.model.decoder(
            trg_positional_encoded,
            e_output,
            e_mask,
            d_mask
        )  # (1, L, d_model)
        output = self.model.softmax(
            self.model.output_linear(decoder_output)
        )  # (1, L, trg_vocab_size)
        output = torch.argmax(output, dim=-1)  # (1, L)
        last_word_id = output[0][i].item()
        
        if i < seq_len-1:
            last_words[i+1] = last_word_id
            cur_len += 1
        
        if last_word_id == eos_id:
            break
    if last_words[-1].item() == pad_id:
        decoded_output = last_words[1:cur_len].tolist()
    else:
        decoded_output = last_words[1:].tolist()
    decoded_output = trg_sp.decode_ids(decoded_output)
    
    return decoded_output

def beam_search(self, e_output, e_mask, trg_sp):
    cur_queue = PriorityQueue()
    for k in range(beam_size):
        cur_queue.put(BeamNode(sos_id, -0.0, [sos_id]))
    
    finished_count = 0
    
    for pos in range(seq_len):
        new_queue = PriorityQueue()
        for k in range(beam_size):
            node = cur_queue.get()
            if node.is_finished:
                new_queue.put(node)
            else:
                trg_input = torch.LongTensor(node.decoded + [pad_id] * (seq_len - len(node.decoded))).to(device)  # (L)
                d_mask = (trg_input.unsqueeze(0) != pad_id).unsqueeze(1).to(device)  # (1, 1, L)
                nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool).to(device)
                nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L)
                d_mask = d_mask & nopeak_mask  # (1, L, L)
                
                trg_embedded = self.model.trg_embedding(trg_input.unsqueeze(0))
                trg_positional_encoded = self.model.positional_encoder(trg_embedded)
                decoder_output = self.model.decoder(
                    trg_positional_encoded,
                    e_output,
                    e_mask,
                    d_mask
                )  # (1, L, d_model)
                output = self.model.softmax(
                    self.model.output_linear(decoder_output)
                )  # (1, L, trg_vocab_size)
                
                output = torch.topk(output[0][pos], dim=-1, k=beam_size)
                last_word_ids = output.indices.tolist()  # (k)
                last_word_prob = output.values.tolist()  # (k)
                
                for i, idx in enumerate(last_word_ids):
                    new_node = BeamNode(idx, -(-node.prob + last_word_prob[i]), node.decoded + [idx])
                    if idx == eos_id:
                        new_node.prob = new_node.prob / float(len(new_node.decoded))
                        new_node.is_finished = True
                        finished_count += 1
                    new_queue.put(new_node)
        
        cur_queue = copy.deepcopy(new_queue)
        
        if finished_count == beam_size:
            break
    
    decoded_output = cur_queue.get().decoded
    
    if decoded_output[-1] == eos_id:
        decoded_output = decoded_output[1:-1]
    else:
        decoded_output = decoded_output[1:]
        
    return trg_sp.decode_ids(decoded_output)
    
def make_mask(self, src_input, trg_input):
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)
    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L)
    d_mask = d_mask & nopeak_mask  # (B, L, L)
    return e_mask, d_mask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help="Checkpoint file name to use for inference")
    parser.add_argument('--input', type=str, required=True, help="Input sentence to translate")
    parser.add_argument('--decode', type=str, required=True, default="greedy", help="Decoding method: 'greedy' or 'beam'")

    args = parser.parse_args()
    
    # Create the Manager instance for inference.
    manager = Manager(is_train=False, ckpt_name=args.model)
    
    # Run inference.
    manager.inference(args.input, args.decode)
