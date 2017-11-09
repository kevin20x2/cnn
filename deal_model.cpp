#define CPU_ONLY
#include "caffe/caffe.hpp"

#include <string>
#include <vector>
#include <iostream>

using namespace caffe;
//#undef NDEBUG
//char * proto = "/home/ftp/VGG_ILSVRC_19_layers_deploy.prototxt";
char * proto = "VGG_ILSVRC_19_layers_deploy.prototxt";
char * model = "VGG_ILSVRC_19_layers.caffemodel";
Phase phase = TEST;
int main()
{

	Caffe::set_mode(Caffe::CPU);
	boost::shared_ptr < Net<float> > net(new caffe::Net<float>(proto,phase));
	//net->CopyTrainedLayersFrom(model);
	NetParameter param;
	ReadNetParamsFromBinaryFileOrDie(model,&param);
    param.mutable_state()->set_phase(TEST);
    param.mutable_state()->set_level(0);
   // param
    //boost::shared_ptr < Net<float> > net(new caffe::Net<float>(param));
    //net->CopyTrainedLayersFrom(param);
    //net->Init(param);
	int num_layers = param.layer_size();

    auto layer_list = net->layers();

    for(int i =0;i<num_layers;++i)
    {
        DLOG(ERROR)<<"Layer" <<i<<":" <<param.layer(i).name()<<"\t"
            << param.layer(i).type();
        if(param.layer(i).type() == "Convolution")
        {
            ConvolutionParameter conv_param = param.layer(i).convolution_param();
            LOG(ERROR)<<"kernel size number"<<conv_param.kernel_size().size();
            int blob_num = param.layer(i).blobs_size();
        //    LOG(ERROR)<<"blobs size " << blob_num;
            //layer_list[i]->blobs().resize(blob_num);
            for(int j = 0;j<blob_num;++j)
            {
				LOG(ERROR) <<"num"<<param.layer(i).blobs(j).num() <<
				   "channel"<<param.layer(i).blobs(j).channels()<<
			    "height" <<param.layer(i).blobs(j).height()<<"width" <<
		      param.layer(i).blobs(j).width();
                //Blob <float > * weights;
                //if( layer_list[i]->blobs().size()>0) {
                  //  weights = layer_list[i]->blobs()[0].get();
               // layer_list[i]->blobs()[j].reset( new Blob<float>());
                //layer_list[i]->blobs()[j]->FromProto(param.layer(i).blobs(j));
               // weights = layer_list[i]->blobs()[j].get();

               /* LOG(ERROR) <<"shape :" <<weights->shape(0)
                           <<" " << weights->shape(1)
                        <<" " << weights->shape(2)
                        <<" " << weights->shape(3);
						*/
               // }

                LOG(ERROR)<<"blob data size:"<<j<<" "<<param.layer(i).blobs(j).data().size();
            }

        }
    }

    return 0;
}
