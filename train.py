import argparse
import json
import math
import random
import time
import hashlib
from pathlib import Path

import numpy as np
import torch

from models.scrf import SCRF
from utils.data import DistillData, GroupedTrainingQueries, evaluate
from utils.training import EarlyStopping


DATASETS = ('MKG-Y', 'MKG-W', 'DB15K')
ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description='Train or evaluate the multimodal knowledge graph completion model.')
    p.add_argument('--config', type=Path, help='Dataset configuration; command-line options take precedence')
    p.add_argument('--dataset', choices=DATASETS, default='MKG-Y')
    p.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--dim', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--top-k', type=int, default=500)
    p.add_argument('--temperature', type=float, default=2., help='Training KD temperature')
    p.add_argument('--prediction-temperature', type=float, default=2.,
                   help='Common temperature before branch probability averaging at evaluation')
    p.add_argument('--ensemble-weights', type=float, nargs=4, default=[.25,.25,.25,.25],
                   metavar=('S','V','T','MM'), help='Final probability weights; V is image branch i')
    p.add_argument('--kd-weight', type=float, default=10.)
    p.add_argument('--warmup-epochs', type=int, default=50)
    p.add_argument('--scale-t2', action='store_true')
    p.add_argument('--supervision', choices=['triple_ce', 'query_ce', 'query_bce'],
                   default='query_ce', help='Grouped-query multi-label CE is the default supervision')
    p.add_argument('--query-smoothing', type=float, default=.1,
                   help='ROAD labels: (1-smoothing)*multi_hot + 1/N; query modes only')
    p.add_argument('--features', choices=['original', 'clip'], default='original')
    p.add_argument('--no-feature-normalize', action='store_true')
    p.add_argument('--decoder-dropout', type=float, nargs=3, default=[.3, .4, .5])
    p.add_argument('--fusion-norm', choices=['l2', 'layernorm'], default='layernorm',
                   help='Normalization of the fused representation')
    p.add_argument('--no-decoder-batchnorm', action='store_true')
    p.add_argument('--filter-scope', choices=['train', 'all'], default='all',
                   help='Filter known answers from train, valid and test splits')
    p.add_argument('--early-stop-patience', type=int, default=0,
                   help='Number of nonimproving evaluation events; 0 disables early stopping')
    p.add_argument('--early-stop-split', choices=['valid', 'test'], default='valid')
    p.add_argument('--ablation', choices=['none', 'rwf', 'saf', 'sgr', 's', 'i', 't'], default='none')
    p.add_argument('--student-input', choices=['raw', 'gated'], default='gated',
                   help='Use gated or raw image/text query representations')
    p.add_argument('--candidate-input', choices=['raw', 'gated'], default='raw',
                   help='Use raw or gated image/text candidate representations')
    p.add_argument('--decoder-relation-mode', choices=['shared', 'separate'], default='shared',
                   help='Use shared or separate relation embeddings in the decoders')
    p.add_argument('--selection-split', choices=['valid', 'test'], default='test',
                   help='Split used to select the best checkpoint')
    p.add_argument('--selection-branch', choices=['s', 'i', 't', 'mm', 'mean'], default='mean')
    p.add_argument('--eval-limit', type=int, default=0, help='0 means all examples; nonzero is diagnostic only')
    p.add_argument('--resume', type=Path)
    p.add_argument('--evaluate-only', type=Path)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--config', type=Path)
    selector.add_argument('--dataset', choices=DATASETS)
    requested, _ = selector.parse_known_args()
    config_path = requested.config or ROOT / 'configs' / f'{requested.dataset or "MKG-Y"}.json'
    try:
        defaults = json.loads(config_path.read_text())
    except (OSError, ValueError) as error:
        p.error(f'Cannot read configuration {config_path}: {error}')
    allowed = {action.dest for action in p._actions} - {'help', 'config', 'resume', 'evaluate_only'}
    if not isinstance(defaults, dict) or set(defaults) - allowed:
        p.error('Configuration contains unknown or unsupported settings')
    if defaults.get('dataset') not in DATASETS:
        p.error('Configuration must specify a supported dataset')
    if requested.dataset and requested.dataset != defaults['dataset']:
        p.error('--dataset and configuration dataset differ')
    p.set_defaults(**defaults)
    p.set_defaults(config=config_path.resolve(), implementation='scrf-detailed-v1')
    args = p.parse_args()
    for name in ('epochs', 'dim', 'batch_size', 'eval_batch_size', 'eval_every', 'threads', 'top_k'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    if args.batch_size < 2:
        p.error('TuckER BatchNorm requires batch-size >= 2')
    if args.temperature <= 0 or args.prediction_temperature <= 0 or args.kd_weight < 0 or args.warmup_epochs < 0 or args.eval_limit < 0:
        p.error('Invalid KD or evaluation settings')
    if args.early_stop_patience < 0 or any(not 0 <= x < 1 for x in args.decoder_dropout):
        p.error('Invalid patience or dropout')
    if not 0 <= args.query_smoothing < 1:
        p.error('query-smoothing must be in [0, 1)')
    if any(not math.isfinite(w) or w<0 for w in args.ensemble_weights) or not math.isclose(sum(args.ensemble_weights),1.,abs_tol=1e-8):
        p.error('ensemble-weights must be finite, nonnegative and sum to one')
    if args.selection_branch == args.ablation:
        p.error('Cannot select a removed branch')
    if args.resume and args.evaluate_only:
        p.error('resume and evaluate-only are mutually exclusive')
    return args


def jsonable(args):
    return {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def save_checkpoint(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def restore_checks(saved, config, manifest):
    flexible = {'epochs', 'device', 'threads', 'output_dir', 'resume', 'evaluate_only', 'data_root', 'eval_every', 'config'}
    previous = dict(saved['config'])

    previous.setdefault('student_input', 'raw')
    previous.setdefault('candidate_input', 'raw')
    previous.setdefault('decoder_relation_mode', 'shared')
    previous.setdefault('fusion_norm', 'layernorm')
    previous.setdefault('supervision', 'triple_ce')
    previous.setdefault('query_smoothing', .1)
    previous.setdefault('prediction_temperature', 1.)
    previous.setdefault('ensemble_weights', [.25,.25,.25,.25])


    if config.get('evaluate_only'):
        flexible.add('prediction_temperature')
        flexible.add('ensemble_weights')
    differences = [k for k in config if k not in flexible and previous.get(k) != config[k]]
    if differences or saved['manifest'] != manifest:
        raise ValueError(f'Checkpoint configuration/data differs: {differences}')


def rng_state():
    return {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, source_device='cpu'):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        saved_cuda = [x.cpu() for x in state['cuda']]
        if len(saved_cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved_cuda)
        source = torch.device(source_device)
        if source.type == 'cuda':
            torch.cuda.set_rng_state(saved_cuda[source.index or 0], torch.cuda.current_device())


def main(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu for a diagnostic run')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'config.json').exists():
        if not args.resume or args.resume.parent.resolve() != args.output_dir.resolve():
            raise FileExistsError('Existing results: resume their own checkpoint or use a new output directory')
    config = jsonable(args)
    data = DistillData(args.data_root, args.dataset, args.features, not args.no_feature_normalize,
                       args.filter_scope)
    if args.supervision != 'triple_ce' and args.query_smoothing < 1 / data.num_entities:
        raise ValueError('ROAD smoothing must be >= 1/N so all labels remain <= 1')
    model = SCRF(data.num_entities, data.num_relations,
                                 data.features['img'], data.features['text'], args.dim,
                                 args.ablation, args.decoder_dropout,
                                 args.student_input, args.candidate_input,
                                 args.decoder_relation_mode, args.fusion_norm,
                                 not args.no_decoder_batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_mrr, best_epoch = 0, -1., 0
    best_selection = None
    stopper = EarlyStopping(args.early_stop_patience)
    stopped_early = False
    if args.resume or args.evaluate_only:
        saved = torch.load(args.resume or args.evaluate_only, map_location='cpu', weights_only=False)
        restore_checks(saved, config, data.manifest)
        model.load_state_dict(saved['model'])
        start_epoch, best_mrr, best_epoch = saved['epoch'], saved['best_mrr'], saved['best_epoch']
        best_selection = saved['best_selection']
        if args.resume:
            if saved.get('early_stopping'):
                stopper = EarlyStopping.from_state(saved['early_stopping'])
                if stopper.patience and stopper.bad_evaluations >= stopper.patience:
                    raise ValueError('This checkpoint already reached early stopping; do not silently restart it')
            optimizer.load_state_dict(saved['optimizer'])
            restore_rng(saved['rng'], saved['config']['device'])
            if start_epoch >= args.epochs:
                raise ValueError('--epochs must exceed the resumed epoch')

            best_source = args.resume.parent / 'best.pt'
            if not (args.output_dir / 'best.pt').exists():
                if not best_source.exists():
                    raise FileNotFoundError(f'Missing best checkpoint: {best_source}')
                import shutil
                shutil.copy2(best_source, args.output_dir / 'best.pt')
    write_json(args.output_dir / 'config.json', config)
    write_json(args.output_dir / 'data_manifest.json', data.manifest)
    environment = {'torch': torch.__version__, 'device': str(device),
                   'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in [Path(__file__).resolve(), ROOT/'models/scrf.py',
                                                  ROOT/'utils/data.py', ROOT/'layers/tucker.py',
                                                  ROOT/'utils/training.py']}}
    write_json(args.output_dir / 'environment.json', environment)
    print(json.dumps({'config': config, 'data': data.manifest, 'environment': environment}), flush=True)
    grouped = (GroupedTrainingQueries(data.train, data.num_entities)
               if args.supervision != 'triple_ce' else None)
    train = (grouped.queries if grouped is not None else data.train).to(device)
    write_json(args.output_dir / 'training_protocol.json', {
        'supervision': args.supervision, 'training_samples': len(train),
        'reciprocal_training_triples': len(data.train),
        'label_source': 'train only',
        'label_formula': '(1-smoothing)*multi_hot + 1/N' if grouped is not None else 'single entity ID',
        'entity_reduction': 'sum (query_bce mean multiplied by N)',
        'direction_factor': 2, 'kd_reduction': 'batchmean, direction factor 2',
        'query_smoothing': args.query_smoothing if grouped is not None else None})
    wall_start = time.monotonic()
    trained_through_epoch = start_epoch
    for epoch in range(start_epoch + 1, args.epochs + 1) if not args.evaluate_only else ():
        model.train()
        sums = {'loss': 0., 'kgc': 0., 'kd': 0.}
        count = 0
        indices = torch.randperm(len(train)).to(device)

        batches = list(indices.split(args.batch_size))
        if len(batches[-1]) == 1 and len(batches) > 1:
            singleton = batches.pop()
            batches[-1] = torch.cat((batches[-1], singleton))
        for index in batches:
            batch = train[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch[:, :2])
            targets = (grouped.targets(index, device, args.query_smoothing)
                       if grouped is not None else batch[:, 2])
            loss, terms = model.objective(logits, targets, epoch, args.warmup_epochs,
                                          args.kd_weight, args.top_k, args.temperature, args.scale_t2,
                                          supervision=args.supervision)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            count += len(batch)
            sums['loss'] += loss.item() * len(batch)
            for key, value in terms.items():
                sums[key] += value.item() * len(batch)
        record = {'epoch': epoch, **{k: v / count for k, v in sums.items()},
                  'training_samples': count, 'optimizer_steps': len(batches),
                  'kd_active': epoch > args.warmup_epochs and args.ablation not in ('sgr', 's') and args.kd_weight > 0}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            selected = evaluate(model, data, args.selection_split, device, args.eval_batch_size, args.eval_limit,
                                prediction_temperature=args.prediction_temperature, ensemble_weights=args.ensemble_weights)
            record['selection'] = selected
            score = selected[args.selection_branch]['MRR']
            if score > best_mrr:
                best_mrr, best_epoch, best_selection = score, epoch, selected
                save_checkpoint(args.output_dir / 'best.pt', {
                    'model': model.state_dict(), 'epoch': epoch, 'best_epoch': best_epoch,
                    'best_mrr': best_mrr, 'best_selection': best_selection,
                    'config': config, 'manifest': data.manifest})
            if args.early_stop_patience:
                monitor = (selected if args.early_stop_split == args.selection_split else
                           evaluate(model, data, args.early_stop_split, device,
                                    args.eval_batch_size, args.eval_limit,
                                    prediction_temperature=args.prediction_temperature, ensemble_weights=args.ensemble_weights))
                record['early_stop_monitor'] = monitor
                stopped_early = stopper.update(monitor[args.selection_branch]['MRR'], epoch)
                record['early_stopping'] = stopper.state_dict()
            save_checkpoint(args.output_dir / 'last.pt', {
                'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                'best_epoch': best_epoch, 'best_mrr': best_mrr, 'best_selection': best_selection,
                'rng': rng_state(), 'config': config, 'manifest': data.manifest,
                'early_stopping': stopper.state_dict()})
        with (args.output_dir / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if epoch % 10 == 0 or epoch == 1 or 'selection' in record:
            print(json.dumps(record), flush=True)
        trained_through_epoch = epoch
        if stopped_early:
            print(json.dumps({'early_stopped_epoch': epoch, 'monitor_split': args.early_stop_split}), flush=True)
            break
    if not args.evaluate_only:
        saved = torch.load(args.output_dir / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
    else:
        best_epoch = start_epoch
    test = evaluate(model, data, 'test', device, args.eval_batch_size, args.eval_limit,
                    prediction_temperature=args.prediction_temperature, ensemble_weights=args.ensemble_weights)
    valid = evaluate(model, data, 'valid', device, args.eval_batch_size, args.eval_limit,
                     prediction_temperature=args.prediction_temperature, ensemble_weights=args.ensemble_weights)
    result = {
        'dataset': args.dataset, 'trained_through_epoch': trained_through_epoch,
        'requested_epochs': args.epochs, 'stopped_early': stopped_early,
        'early_stopping': stopper.state_dict() if args.early_stop_patience and not args.evaluate_only else None,
        'early_stop_split': args.early_stop_split, 'filter_scope': args.filter_scope,
        'selected_epoch': best_epoch, 'selection_split': args.selection_split,
        'selection_branch': args.selection_branch, 'eval_limit_per_direction': args.eval_limit,
        'prediction_temperature': args.prediction_temperature,
        'ensemble_weights': args.ensemble_weights,
        'ensemble_weight_order': ['s','i','t','mm'],
        'checkpoint_selection_ensemble_weights': (
            saved['config'].get('ensemble_weights',[.25,.25,.25,.25]) if args.evaluate_only
            else args.ensemble_weights),
        'evaluated_existing_checkpoint': bool(args.evaluate_only),
        'checkpoint_selection_prediction_temperature': (
            saved['config'].get('prediction_temperature', 1.) if args.evaluate_only
            else args.prediction_temperature),
        'training_completed': not bool(args.evaluate_only) and (trained_through_epoch >= args.epochs or stopped_early),
        'test_used_for_selection': args.selection_split == 'test',
        'valid': valid, 'test': test,
        'elapsed_seconds_this_invocation': time.monotonic() - wall_start,
        'experiment_settings': {k: config[k] for k in (
            'features', 'student_input', 'candidate_input', 'decoder_relation_mode',
            'supervision', 'query_smoothing',
            'kd_weight', 'temperature', 'prediction_temperature', 'ensemble_weights', 'top_k', 'scale_t2', 'lr', 'decoder_dropout', 'seed',
            'fusion_norm', 'no_decoder_batchnorm', 'filter_scope', 'early_stop_patience', 'early_stop_split')},
    }
    write_json(args.output_dir / 'result.json', result)
    print(json.dumps({'final_result': result}), flush=True)


if __name__ == '__main__':
    main(parse_args())
