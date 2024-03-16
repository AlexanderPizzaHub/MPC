#include<iostream>
#include"Eigen/Dense"
using namespace Eigen;
/*
The controller class for linear control.
*/
class Controller{
    public:

        void setup_dynamics(MatrixXd A, MatrixXd B, VectorXd f);
        void setup_cost(MatrixXd Q, MatrixXd R, MatrixXd F);
        void setup_dimension(int state_dim,int control_dim);
        void discretize(double dt);

    private:
        MatrixXd A; 
        MatrixXd B; 

        VectorXd f;

        MatrixXd Q;
        MatrixXd R;
        MatrixXd F;

        MatrixXd Q_bar;
        MatrixXd R_bar;
        MatrixXd G;
        MatrixXd M;

        int control_horizon;
        int pred_horizon;
        int state_dim;
        int control_dim;

};

void Controller::setup_dynamics(MatrixXd A, MatrixXd B, VectorXd f){
    this->A = A;
    this->B = B;
    this->f = f;
}

void Controller::setup_dimension(int state_dim,int control_dim){
    this->state_dim = state_dim;
    this->control_dim = control_dim;
}

void Controller::setup_cost(MatrixXd Q, MatrixXd R, MatrixXd F){
    this->Q = Q;
    this->R = R;
    this->F = F;
}

void Controller::discretize(double dt){
        G = MatrixXd::Zero(state_dim*pred_horizon,control_dim*pred_horizon);
        MatrixXd N = MatrixXd::Zero((pred_horizon+1)*state_dim,(pred_horizon+1)*state_dim);

        MatrixXd tmp_B = dt * B;

        MatrixXd del_A = MatrixXd::Identity(state_dim,state_dim) + dt * A;

        MatrixXd cumprod_A = del_A; 

        /*
        TODO: Implement the standard discrete time LQR setup
        */

       N.block(0,0,state_dim-1,state_dim-1) = MatrixXd::Identity(state_dim,state_dim);

       for(int i=1;i<pred_horizon+1;i++){ 
        N.block(i*state_dim,i*state_dim,(i+1)*state_dim-1,(i+1)*state_dim-1) = cumprod_A;
        for(int j=0;j<pred_horizon;j++){
            /*
            Think carefully about the range of i and j.
            */
            G.block((j+i)*state_dim,j*control_dim,(j+i+1)*state_dim,(j+1)*control_dim) = cumprod_A * tmp_B;
        }
        cumprod_A *= del_A;
       }

        Q_bar = MatrixXd::Zero(pred_horizon*state_dim,pred_horizon*control_dim);


        Q_bar.block(0,0,(pred_horizon-1)*state_dim,(pred_horizon-1)*state_dim) = MatrixXd::Identity(state_dim,state_dim);

        for(int i=1;i<pred_horizon-1;i++){
            Q_bar.block(i*state_dim,i*state_dim,(i+1)*state_dim-1,(i+1)*state_dim-1) = Q;
        }

        Q_bar.block((pred_horizon-1)*state_dim,(pred_horizon-1)*state_dim,pred_horizon*state_dim-1,pred_horizon*state_dim-1) = F;

        R_bar = MatrixXd::Zero(pred_horizon*control_dim,pred_horizon*control_dim);
        for(int i=0;i<pred_horizon;i++){
            R_bar.block(i*control_dim,i*control_dim,(i+1)*control_dim-1,(i+1)*control_dim-1) = R;
        }

        M = G.adjoint() * Q_bar * G + R_bar;

}

int main(){
    Controller ctrl;
    return 0;
}


