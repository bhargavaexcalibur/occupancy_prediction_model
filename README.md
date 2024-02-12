# Interview challenge 


# 1) Occupancy prediction model

### Submission details
I have tried two solutions for solving the occupany model. 
1. Random Forest 
2. Prophet model 

The pre processing steps for both these models is almost the same. 
RF model beats prophet model , so please use Random Forest model for the final evaluation.

To get the random forest results please run:
python3 ./Rf_solution.py '2022-08-31 23:59:59' data/device_activations.csv RF_result.csv

To get the prophet results please run:
python3 ./bh_propht.py '2022-08-31 23:59:59' data/device_activations.csv prophet_result.csv

# 2) Bonus: Dockerized REST API

I have containerised the Random Forest solution and could not containerise the prophet solution in time. 
Instead of the docker build and docker run, I have used docker init and docker compose.

Currently the submission includes the port 8090. If you want to change it 
please run : docker init 
and it will create the necessary input files / docker files. 
These files are already created, do run docker init if you want to change the port.
after docker init you would be given a set of options :

Do you want to overwrite them? Yes
? What application platform does your project use? Python
? What version of Python do you want to use? 3.10.0
? What port do you want your app to listen on? 8090
? What is the command to run your app? gunicorn 'app:app' --bind=0.0.0.0:8090

After running docker init, please run "docker compose up --build" and application will be available at http://localhost:8090




Example usage of your Docker application provided by zscaler.

    docker build -t <your-rest-api-docker-image> .
    docker run -t <your-rest-api-docker-image> -d
    curl http://<DOCKER-IP>:5000/predict

