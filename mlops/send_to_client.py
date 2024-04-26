
       #server url
tracking_server_url = 'http://localhost:8080/'
        
mlflow_client = mlflow.tracking.MlflowClient(tracking_uri=tracking_server_url)
exp_name = mlflow_client.get_experiment_by_name("fatigue_with_CV")
exp_id = exp_name.experiment_id
last_best_run = mlflow_client.search_runs(experiment_ids=[exp_id], order_by=["metrics.val_loss ASC"])[0]
config = json.loads(json.dumps(last_best_run.data.params))
       
#getting parameter values
lr = config['train_loop_config/lr']
batch_size = config['train_loop_config/batch_size']
lr_factor = config['train_loop_config/lr_factor']
fc_size = config['train_loop_config/fc_size']
lr_patience = config['train_loop_config/lr_patience']
num_samples = config['train_loop_config/num_samples']
num_epochs = config['train_loop_config/num_epochs']

print(last_best_run)
print(batch_size)
print(num_epochs)
