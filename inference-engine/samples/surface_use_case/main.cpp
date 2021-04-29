// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <inference_engine.hpp>
#include <vpu/vpu_plugin_config.hpp>
#include <cldnn/cldnn_config.hpp>
#include <gna/gna_config.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "surface_use_case.hpp"
#include "infer_request_wrap.hpp"
#include "progress_bar.hpp"
#include "statistics_report.hpp"
#include "inputs_filling.hpp"
#include "utils.hpp"
//#include "bits/stdc++.h"

#define custom_use_case  1

using namespace InferenceEngine;

static const size_t progressBarDefaultTotalCount = 1000;

uint64_t getDurationInMilliseconds(uint32_t duration) {
    return duration * 1000LL;
}

uint64_t getDurationInNanoseconds(uint32_t duration) {
    return duration * 1000000000LL;
}

template<typename T>
using uniformDistribution =
typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<
    std::is_integral<T>::value,
    std::uniform_int_distribution<T>,
    void>::type
>::type;
template<typename T, typename T2>
void fillBlobRandom( Blob::Ptr& inputBlob,
    T rand_min = std::numeric_limits<T>::min(),
    T rand_max = std::numeric_limits<T>::max() )
{
    MemoryBlob::Ptr minput = as<MemoryBlob>( inputBlob );
    if( !minput )
    {
        IE_THROW() << "We expect inputBlob to be inherited from MemoryBlob in fillBlobRandom, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = minput->wmap();

    auto inputBlobData = minputHolder.as<T*>();
    std::mt19937 gen( 0 );
    uniformDistribution<T2> distribution( rand_min, rand_max );
    for( size_t i = 0; i < inputBlob->size(); i++ )
    {
        inputBlobData[i] = static_cast<T>(distribution( gen ));
    }
}

std::string model_path = "D:\\work\\models\\";
std::vector<std::string> pipe_string = { "mobilenetv2.xml,30,100" , "mobilenetv2.xml,resnet50.xml,30,100" };

struct pipeline_infer
{
    std::vector< blob_stream_t > blob_stream_list;
    std::vector<CNNNetwork> cnn_network_pipe;
    std::vector<ExecutableNetwork> exec_network;
    std::vector<InferRequest> infer_request;
    //std::vector< std::unique_ptr<InferRequestsQueue> > infer_request_queue;
    //infer_request_queue.reserve( blob_stream_list.size() );
    //InferRequestsQueue inferRequestsQueue( exeNetwork, nireq );
    std::vector<surface_use_case::InputsInfo> app_inputs_info;

    //std::vector< std::unique_ptr< pipeline_infer > > _inference_pipe;
        

    uint32_t _current_iteration = 0;
    std::thread _thread;

    void ExecuteThread()
    {
        // Schedule the inferences
        while( _current_iteration < blob_stream_list[0]._iteration_count )
        {
            //_fixture.StartTimer( _latency_timer[_current_iteration] );
            uint32_t counter = 0;
            for( auto& inference : infer_request )
            {
                std::cout << "Blob stream: " << blob_stream_list[counter]._blob << std::endl;
                inference.Infer();
                counter++;
                //inference->Readback( _current_iteration );
            }

            //_fixture.StopTimer( _latency_timer[_current_iteration] );

            _current_iteration++;
        }
    }

    void Execute( void )
    {
        //_fixture.StartTimer( _throughput_timer );

        _thread = std::thread( [&]() { ExecuteThread(); } );
    }

    ///////////////////////////////////////////////////////////////////////
    void WaitForCompletion( void )
    {
        _thread.join();

        //_fixture.StopTimer( _throughput_timer );
    };

    ///////////////////////////////////////////////////////////////////////
    //bool Run( void )
    //{
    //    bool result = true;

    //    try
    //    {
    //        // Schedule the inference streams
    //        for( auto& _inference_stream : _inference_pipe )
    //        {
    //            _inference_stream->Execute();
    //        }

    //        // Wait for them to complete
    //        std::vector< std::thread > threads;
    //        for( auto& _inference_stream : _inference_pipe )
    //        {
    //            threads.emplace_back( std::thread(
    //                [&]() { _inference_stream->WaitForCompletion(); } ) );
    //        }
    //        for( auto& thread : threads )
    //        {
    //            thread.join();
    //        }
    //    }
    //    catch( ... )
    //    {
    //        result = false;
    //    }

    //    return result;
    //}
};



bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validating input arguments--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_help || FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    /*if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }*/
    if( FLAGS_blob_stream.empty() )
    {
        showUsage();
        throw std::logic_error( "Use case blob_stream is required but not set. Please set -blob_stream option." );
    }

    if (FLAGS_api != "async" && FLAGS_api != "sync") {
        throw std::logic_error("Incorrect API. Please set -api option to `sync` or `async` value.");
    }

    if (!FLAGS_report_type.empty() &&
        FLAGS_report_type != noCntReport && FLAGS_report_type != averageCntReport && FLAGS_report_type != detailedCntReport) {
        std::string err = "only " + std::string(noCntReport) + "/" + std::string(averageCntReport) + "/" + std::string(detailedCntReport) +
                          " report types are supported (invalid -report_type option value)";
        throw std::logic_error(err);
    }

    if ((FLAGS_report_type == averageCntReport) && ((FLAGS_d.find("MULTI") != std::string::npos))) {
        throw std::logic_error("only " + std::string(detailedCntReport) + " report type is supported for MULTI device");
    }

    bool isNetworkCompiled = fileExt(FLAGS_m) == "blob";
    bool isPrecisionSet = !(FLAGS_ip.empty() && FLAGS_op.empty() && FLAGS_iop.empty());
    if (isNetworkCompiled && isPrecisionSet) {
        std::string err = std::string("Cannot set precision for a compiled network. ") +
                          std::string("Please re-compile your network with required precision using compile_tool");

        throw std::logic_error(err);
    }
    return true;
}

static void next_step(const std::string additional_info = "") {
    static size_t step_id = 0;
    static const std::map<size_t, std::string> step_names = {
            { 1, "Parsing and validating input arguments" },
            { 2, "Loading Inference Engine" },
            { 3, "Setting device configuration" },
            { 4, "Reading network files" },
            { 5, "Resizing network to match image sizes and given batch" },
            { 6, "Configuring input of the model" },
            { 7, "Loading the model to the device" },
            { 8, "Setting optimal runtime parameters" },
            { 9, "Creating infer requests and filling input blobs with images" },
            { 10, "Measuring performance" },
            { 11, "Dumping statistics report" }
    };

    step_id++;
    if (step_names.count(step_id) == 0)
        IE_THROW() << "Step ID " << step_id << " is out of total steps number " << step_names.size();

    std::cout << "[Step " << step_id << "/" << step_names.size() << "] " << step_names.at(step_id)
              << (additional_info.empty() ? "" : " (" + additional_info + ")") << std::endl;
}

template <typename T>
T getMedianValue(const std::vector<T> &vec) {
    std::vector<T> sortedVec(vec);
    std::sort(sortedVec.begin(), sortedVec.end());
    return (sortedVec.size() % 2 != 0) ?
           sortedVec[sortedVec.size() / 2ULL] :
           (sortedVec[sortedVec.size() / 2ULL] + sortedVec[sortedVec.size() / 2ULL - 1ULL]) / static_cast<T>(2.0);
}

/**
* @brief The entry point of the surface_use_case application
*/
int main(int argc, char *argv[]) {
    std::shared_ptr<StatisticsReport> statistics;
    try
    {
        ExecutableNetwork exeNetwork;

        // ----------------- 1. Parsing and validating input arguments -------------------------------------------------
        next_step();

        if( !ParseAndCheckCommandLine( argc, argv ) )
        {
            return 0;
        }

        bool isNetworkCompiled = fileExt( FLAGS_m ) == "blob";
        if( isNetworkCompiled )
        {
            slog::info << "Network is compiled" << slog::endl;
        }

        std::vector<gflags::CommandLineFlagInfo> flags;
        StatisticsReport::Parameters command_line_arguments;
        gflags::GetAllFlags( &flags );
        for( auto& flag : flags )
        {
            if( !flag.is_default )
            {
                command_line_arguments.push_back( { flag.name, flag.current_value } );
            }
        }
        if( !FLAGS_report_type.empty() )
        {
            statistics = std::make_shared<StatisticsReport>( StatisticsReport::Config{ FLAGS_report_type, FLAGS_report_folder } );
            statistics->addParameters( StatisticsReport::Category::COMMAND_LINE_PARAMETERS, command_line_arguments );
        }
        auto isFlagSetInCommandLine = [&command_line_arguments]( const std::string& name )
        {
            return (std::find_if( command_line_arguments.begin(), command_line_arguments.end(),
                [name]( const std::pair<std::string, std::string>& p ) { return p.first == name; } ) != command_line_arguments.end());
        };

        std::string device_name = FLAGS_d;

        // Parse devices
        auto devices = parseDevices( device_name );

        // Parse nstreams per device
        std::map<std::string, std::string> device_nstreams = parseNStreamsValuePerDevice( devices, FLAGS_nstreams );

        // Load device config file if specified
        std::map<std::string, std::map<std::string, std::string>> config;
#ifdef USE_OPENCV
        if( !FLAGS_load_config.empty() )
        {
            load_config( FLAGS_load_config, config );
        }
#endif
        /** This vector stores paths to the processed images **/
        std::vector<std::string> inputFiles;
        parseInputFilesArguments( inputFiles );

        // ----------------- 2. Loading the Inference Engine -----------------------------------------------------------
        next_step();

        Core ie;
        if( FLAGS_d.find( "CPU" ) != std::string::npos && !FLAGS_l.empty() )
        {
            // CPU (MKLDNN) extensions is loaded as a shared library and passed as a pointer to base extension
            const auto extension_ptr = std::make_shared<InferenceEngine::Extension>( FLAGS_l );
            ie.AddExtension( extension_ptr );
            slog::info << "CPU (MKLDNN) extensions is loaded " << FLAGS_l << slog::endl;
        }

        // Load clDNN Extensions
        if( (FLAGS_d.find( "GPU" ) != std::string::npos) && !FLAGS_c.empty() )
        {
            // Override config if command line parameter is specified
            if( !config.count( "GPU" ) )
                config["GPU"] = {};
            config["GPU"][CONFIG_KEY( CONFIG_FILE )] = FLAGS_c;
        }
        if( config.count( "GPU" ) && config.at( "GPU" ).count( CONFIG_KEY( CONFIG_FILE ) ) )
        {
            auto ext = config.at( "GPU" ).at( CONFIG_KEY( CONFIG_FILE ) );
            ie.SetConfig( { { CONFIG_KEY( CONFIG_FILE ), ext } }, "GPU" );
            slog::info << "GPU extensions is loaded " << ext << slog::endl;
        }

        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions( device_name ) << std::endl;

        // ----------------- 3. Setting device configuration -----------------------------------------------------------
        next_step();

        bool perf_counts = false;
        // Update config per device according to command line parameters
        for( auto& device : devices )
        {
            if( !config.count( device ) ) config[device] = {};
            std::map<std::string, std::string>& device_config = config.at( device );

            // Set performance counter
            if( isFlagSetInCommandLine( "pc" ) )
            {
                // set to user defined value
                device_config[CONFIG_KEY( PERF_COUNT )] = FLAGS_pc ? CONFIG_VALUE( YES ) : CONFIG_VALUE( NO );
            }
            else if( device_config.count( CONFIG_KEY( PERF_COUNT ) ) &&
                (device_config.at( CONFIG_KEY( PERF_COUNT ) ) == "YES") )
            {
                slog::warn << "Performance counters for " << device <<
                    " device is turned on. To print results use -pc option." << slog::endl;
            }
            else if( FLAGS_report_type == detailedCntReport || FLAGS_report_type == averageCntReport )
            {
                slog::warn << "Turn on performance counters for " << device <<
                    " device since report type is " << FLAGS_report_type << "." << slog::endl;
                device_config[CONFIG_KEY( PERF_COUNT )] = CONFIG_VALUE( YES );
            }
            else if( !FLAGS_exec_graph_path.empty() )
            {
                slog::warn << "Turn on performance counters for " << device <<
                    " device due to execution graph dumping." << slog::endl;
                device_config[CONFIG_KEY( PERF_COUNT )] = CONFIG_VALUE( YES );
            }
            else
            {
                // set to default value
                device_config[CONFIG_KEY( PERF_COUNT )] = FLAGS_pc ? CONFIG_VALUE( YES ) : CONFIG_VALUE( NO );
            }
            perf_counts = (device_config.at( CONFIG_KEY( PERF_COUNT ) ) == CONFIG_VALUE( YES )) ? true : perf_counts;

            auto setThroughputStreams = [&]()
            {
                const std::string key = device + "_THROUGHPUT_STREAMS";
                if( device_nstreams.count( device ) )
                {
                    // set to user defined value
                    std::vector<std::string> supported_config_keys = ie.GetMetric( device, METRIC_KEY( SUPPORTED_CONFIG_KEYS ) );
                    if( std::find( supported_config_keys.begin(), supported_config_keys.end(), key ) == supported_config_keys.end() )
                    {
                        throw std::logic_error( "Device " + device + " doesn't support config key '" + key + "'! " +
                            "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>" +
                            " or via configuration file." );
                    }
                    device_config[key] = device_nstreams.at( device );
                }
                else if( !device_config.count( key ) && (FLAGS_api == "async") )
                {
                    slog::warn << "-nstreams default value is determined automatically for " << device << " device. "
                        "Although the automatic selection usually provides a reasonable performance, "
                        "but it still may be non-optimal for some cases, for more information look at README." << slog::endl;
                    if( std::string::npos == device.find( "MYRIAD" ) ) // MYRIAD sets the default number of streams implicitly (without _AUTO)
                        device_config[key] = std::string( device + "_THROUGHPUT_AUTO" );
                }
                if( device_config.count( key ) )
                    device_nstreams[device] = device_config.at( key );
            };

            if( device == "CPU" )
            {  // CPU supports few special performance-oriented keys
// limit threading for CPU portion of inference
                if( isFlagSetInCommandLine( "nthreads" ) )
                    device_config[CONFIG_KEY( CPU_THREADS_NUM )] = std::to_string( FLAGS_nthreads );

                if( isFlagSetInCommandLine( "enforcebf16" ) )
                    device_config[CONFIG_KEY( ENFORCE_BF16 )] = FLAGS_enforcebf16 ? CONFIG_VALUE( YES ) : CONFIG_VALUE( NO );

                if( isFlagSetInCommandLine( "pin" ) )
                {
                    // set to user defined value
                    device_config[CONFIG_KEY( CPU_BIND_THREAD )] = FLAGS_pin;
                }
                else if( !device_config.count( CONFIG_KEY( CPU_BIND_THREAD ) ) )
                {
                    if( (device_name.find( "MULTI" ) != std::string::npos) &&
                        (device_name.find( "GPU" ) != std::string::npos) )
                    {
                        slog::warn << "Turn off threads pinning for " << device <<
                            " device since multi-scenario with GPU device is used." << slog::endl;
                        device_config[CONFIG_KEY( CPU_BIND_THREAD )] = CONFIG_VALUE( NO );
                    }
                    else
                    {
                        // set to default value
                        device_config[CONFIG_KEY( CPU_BIND_THREAD )] = FLAGS_pin;
                    }
                }

                // for CPU execution, more throughput-oriented execution via streams
                setThroughputStreams();
            }
            else if( device == ("GPU") )
            {
                // for GPU execution, more throughput-oriented execution via streams
                setThroughputStreams();

                if( (device_name.find( "MULTI" ) != std::string::npos) &&
                    (device_name.find( "CPU" ) != std::string::npos) )
                {
                    slog::warn << "Turn on GPU trottling. Multi-device execution with the CPU + GPU performs best with GPU trottling hint," <<
                        "which releases another CPU thread (that is otherwise used by the GPU driver for active polling)" << slog::endl;
                    device_config[CLDNN_CONFIG_KEY( PLUGIN_THROTTLE )] = "1";
                }
            }
            else if( device == "MYRIAD" )
            {
                device_config[CONFIG_KEY( LOG_LEVEL )] = CONFIG_VALUE( LOG_WARNING );
                setThroughputStreams();
            }
            else if( device == "GNA" )
            {
                if( FLAGS_qb == 8 )
                    device_config[GNA_CONFIG_KEY( PRECISION )] = "I8";
                else
                    device_config[GNA_CONFIG_KEY( PRECISION )] = "I16";

                if( isFlagSetInCommandLine( "nthreads" ) )
                    device_config[GNA_CONFIG_KEY( LIB_N_THREADS )] = std::to_string( FLAGS_nthreads );
            }
            else
            {
                std::vector<std::string> supported_config_keys = ie.GetMetric( device, METRIC_KEY( SUPPORTED_CONFIG_KEYS ) );
                auto supported = [&]( const std::string& key )
                {
                    return std::find( std::begin( supported_config_keys ), std::end( supported_config_keys ), key )
                        != std::end( supported_config_keys );
                };
                if( supported( CONFIG_KEY( CPU_THREADS_NUM ) ) && isFlagSetInCommandLine( "nthreads" ) )
                {
                    device_config[CONFIG_KEY( CPU_THREADS_NUM )] = std::to_string( FLAGS_nthreads );
                }
                if( supported( CONFIG_KEY( CPU_THROUGHPUT_STREAMS ) ) && isFlagSetInCommandLine( "nstreams" ) )
                {
                    device_config[CONFIG_KEY( CPU_THROUGHPUT_STREAMS )] = FLAGS_nstreams;
                }
                if( supported( CONFIG_KEY( CPU_BIND_THREAD ) ) && isFlagSetInCommandLine( "pin" ) )
                {
                    device_config[CONFIG_KEY( CPU_BIND_THREAD )] = FLAGS_pin;
                }
            }
        }

        for( auto&& item : config )
        {
            ie.SetConfig( item.second, item.first );
        }

        auto double_to_string = []( const double number )
        {
            std::stringstream ss;
            ss << std::fixed << std::setprecision( 2 ) << number;
            return ss.str();
        };
        auto get_total_ms_time = []( Time::time_point& startTime )
        {
            return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
        };

#if custom_use_case
        //Parse blob_stream
        //Blob stream includes 3 items:
        // actual blob used(IRv10 for CPU and *.blob for VPU), target FPS and number of iterations to run. 

        uint32_t num_pipes = 2;
        std::vector<pipeline_infer> pipe_stream;

        

        for(int i=0; i< pipe_string.size(); i++ ){
            std::stringstream parse_vector( pipe_string[i] );
            uint32_t num_blobs = 0, fps_done=0;
            pipe_stream.emplace_back( pipeline_infer() );
            while( parse_vector.good() )
            {
                std::string substr;
                getline( parse_vector, substr, ',' );
                pipe_stream[i].blob_stream_list.emplace_back( blob_stream_t() );
                if( std::isdigit( substr.at(0) ) == 0 )
                {
                    pipe_stream[i].blob_stream_list[num_blobs]._blob = model_path + substr;
                    num_blobs++;
                }
                else if( (std::isdigit( substr.at( 0 ) ) != 0) && (fps_done == 0) )
                {
                    for( uint32_t numblob=0; numblob < num_blobs; numblob++ )
                    {
                        pipe_stream[i].blob_stream_list[numblob]._target_fps = std::stoi( substr );                        
                    }
                    fps_done = 1;
                }
                else if( (std::isdigit( substr.at( 0 ) ) != 0) && (fps_done == 1) )
                {
                    for( uint32_t numblob = 0; numblob < num_blobs; numblob++ )
                    {
                        pipe_stream[i].blob_stream_list[numblob]._iteration_count = std::stoi( substr );
                        pipe_stream[i].blob_stream_list[numblob].pipeline = i;
                    }
                    
                }
                                
            }

            

            
            
        }

        //pipe_stream.blob_stream_list = parseblobstream( FLAGS_blob_stream );

        //
        /*std::vector< std::string > all_blobs;
        std::string model_path = "D:\\work\\models\\";
        for( int i = 0; i < pipe_stream.blob_stream_list.size(); i++ )
        {
            all_blobs.emplace_back( std::string() );
            all_blobs[i] = model_path + blob_stream_list[i]._blob;
        }*/

        size_t batchSize = FLAGS_b;
        Precision precision = Precision::UNSPECIFIED;
        std::string topology_name = "";
        //surface_use_case::InputsInfo app_inputs_info;
        std::string output_name;

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if( nireq == 0 )
        {
            if( FLAGS_api == "sync" )
            {
                nireq = 1;
            }
            else
            {
                /*std::string key = METRIC_KEY( OPTIMAL_NUMBER_OF_INFER_REQUESTS );
                try
                {
                    nireq = exeNetwork.GetMetric( key ).as<unsigned int>();
                }
                catch( const std::exception& ex )
                {
                    IE_THROW()
                        << "Every device used with the surface_use_case should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << device_name << " with error:" << ex.what();
                }*/
            }
        }

                 

        if( !isNetworkCompiled )
        {
            // ----------------- 4. Reading the Intermediate Representation network ----------------------------------------
            next_step();

            slog::info << "Loading network files" << slog::endl;

            for( int i = 0; i < pipe_string.size(); i++ ){

                for( int j = 0; j < pipe_stream[i].blob_stream_list.size(); j++ )
                {
                    if( pipe_stream[i].blob_stream_list[j]._blob.empty() )
                    {
                        break;
                    }
                    pipe_stream[i].cnn_network_pipe.emplace_back( CNNNetwork() );
                    pipe_stream[i].exec_network.emplace_back( ExecutableNetwork() );
                    pipe_stream[i].infer_request.emplace_back( InferRequest() );

                    pipe_stream[i].app_inputs_info.emplace_back( surface_use_case::InputsInfo() );
                    //auto startTime = Time::now();
                    pipe_stream[i].cnn_network_pipe[j] = ie.ReadNetwork( pipe_stream[i].blob_stream_list[j]._blob );

                    //////////////////////////////////////////////////
                    const InputsDataMap inputInfo( pipe_stream[i].cnn_network_pipe[j].getInputsInfo() );
                    bool reshape = false;
                    pipe_stream[i].app_inputs_info[j] = getInputsInfo<InputInfo::Ptr>( FLAGS_shape, FLAGS_layout, FLAGS_b, inputInfo, reshape );
                    if( reshape )
                    {
                        InferenceEngine::ICNNNetwork::InputShapes shapes = {};
                        for( auto& item : pipe_stream[i].app_inputs_info[j] )
                            shapes[item.first] = item.second.shape;
                        slog::info << "Reshaping network: " << getShapesString( shapes ) << slog::endl;
                        auto startTime = Time::now();
                        pipe_stream[i].cnn_network_pipe[j].reshape( shapes );
                        auto duration_ms = double_to_string( get_total_ms_time( startTime ) );
                        slog::info << "Reshape network took " << duration_ms << " ms" << slog::endl;
                        if( statistics )
                            statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                                {
                                        {"reshape network time (ms)", duration_ms}
                                } );
                    }

                    /////////////////////////////////////////////////////

                    processPrecision( pipe_stream[i].cnn_network_pipe[j], FLAGS_ip, FLAGS_op, FLAGS_iop );
                    for( auto& item : pipe_stream[i].cnn_network_pipe[j].getInputsInfo() )
                    {
                        // if precision for input set by user, then set it to app_inputs
                        // if it an image, set U8
                        if( !FLAGS_ip.empty() || FLAGS_iop.find( item.first ) != std::string::npos )
                        {
                            pipe_stream[i].app_inputs_info[j].at( item.first ).precision = item.second->getPrecision();
                        }
                        else if( pipe_stream[i].app_inputs_info[j].at( item.first ).isImage() )
                        {
                            pipe_stream[i].app_inputs_info[j].at( item.first ).precision = Precision::U8;
                            item.second->setPrecision( pipe_stream[i].app_inputs_info[j].at( item.first ).precision );
                        }
                    }

                    printInputAndOutputsInfo( pipe_stream[i].cnn_network_pipe[j] );
                    // ----------------- 7. Loading the model to the device --------------------------------------------------------
                    next_step();
                    //startTime = Time::now();
                    std::string input_name = pipe_stream[i].cnn_network_pipe[j].getInputsInfo().begin()->first;
                    pipe_stream[i].exec_network[j] = ie.LoadNetwork( pipe_stream[i].cnn_network_pipe[j], device_name );

                    pipe_stream[i].infer_request[j] = pipe_stream[i].exec_network[j].CreateInferRequest();


                    Blob::Ptr inputBlob;
                    for( auto& item : pipe_stream[i].app_inputs_info[j] )
                    {
                        inputBlob = pipe_stream[i].infer_request[j].GetBlob( item.first );
                        auto app_info = pipe_stream[i].app_inputs_info[j].at( item.first );
                        auto precision = app_info.precision;

                        // Fill random
                        slog::info << "Fill input '" << item.first << "' with random values ("
                            << std::string( (app_info.isImage() ? "image" : "some binary data") )
                            << " is expected)" << slog::endl;
                        if( precision == InferenceEngine::Precision::FP32 )
                        {
                            fillBlobRandom<float, float>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::FP16 )
                        {
                            fillBlobRandom<short, short>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::I32 )
                        {
                            fillBlobRandom<int32_t, int32_t>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::I64 )
                        {
                            fillBlobRandom<int64_t, int64_t>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::U8 )
                        {
                            // uniform_int_distribution<uint8_t> is not allowed in the C++17 standard and vs2017/19
                            fillBlobRandom<uint8_t, uint32_t>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::I8 )
                        {
                            // uniform_int_distribution<int8_t> is not allowed in the C++17 standard and vs2017/19
                            fillBlobRandom<int8_t, int32_t>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::U16 )
                        {
                            fillBlobRandom<uint16_t, uint16_t>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::I16 )
                        {
                            fillBlobRandom<int16_t, int16_t>( inputBlob );
                        }
                        else if( precision == InferenceEngine::Precision::BOOL )
                        {
                            fillBlobRandom<uint8_t, uint32_t>( inputBlob, 0, 1 );
                        }
                        else
                        {
                            IE_THROW() << "Input precision is not supported for " << item.first;
                        }
                    }

                    pipe_stream[i].infer_request[j].SetBlob( input_name, inputBlob );
                }


                //Blob::Ptr inputBlob = infer_request[i].SetBlob(); 
                //InferRequestsQueue tmp_inferrequestqueue( exec_network[i], nireq );
                //infer_request_queue[i] = std::make_unique< InferRequestsQueue >( exec_network[i], nireq );
                
                //fillBlobs( inputFiles, batchSize = 1, app_inputs_info[i], infer_request[i]);
            }
            

        }

        // --------------------------- Step 7. Do inference --------------------------------------------------------
        /* Running the pipelines via multiple threads */
        
        for( int i = 0; i < pipe_string.size(); i++ )
        {
            //_inference_pipe = pipe_stream[i]._inference_pipe;
            pipe_stream[i].Execute();
            std::cout << "Pipe Number: " << i << std::endl;

        }

        // Wait for them to complete
        std::vector< std::thread > threads;
        for( auto& _inference_stream : pipe_stream )
        {
            threads.emplace_back( std::thread(
                [&]() { _inference_stream.WaitForCompletion(); } ) );
        }
        for( auto& thread : threads )
        {
            thread.join();
        }
                
        std::cout << "Execution Complete" << std::endl;
        // -----------------------------------------------------------------------------------------------------

        


#else
        size_t batchSize = FLAGS_b;
        Precision precision = Precision::UNSPECIFIED;
        std::string topology_name = "";
        surface_use_case::InputsInfo app_inputs_info;
        std::string output_name;
        if( !isNetworkCompiled )
        {
            // ----------------- 4. Reading the Intermediate Representation network ----------------------------------------
            next_step();

            slog::info << "Loading network files" << slog::endl;

            auto startTime = Time::now();
            CNNNetwork cnnNetwork = ie.ReadNetwork( FLAGS_m );
            auto duration_ms = double_to_string( get_total_ms_time( startTime ) );
            slog::info << "Read network took " << duration_ms << " ms" << slog::endl;
            if( statistics )
                statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                    {
                            {"read network time (ms)", duration_ms}
                    } );

            const InputsDataMap inputInfo( cnnNetwork.getInputsInfo() );
            if( inputInfo.empty() )
            {
                throw std::logic_error( "no inputs info is provided" );
            }

            // ----------------- 5. Resizing network to match image sizes and given batch ----------------------------------
            next_step();
            batchSize = cnnNetwork.getBatchSize();
            // Parse input shapes if specified
            bool reshape = false;
            app_inputs_info = getInputsInfo<InputInfo::Ptr>( FLAGS_shape, FLAGS_layout, FLAGS_b, inputInfo, reshape );
            if( reshape )
            {
                InferenceEngine::ICNNNetwork::InputShapes shapes = {};
                for( auto& item : app_inputs_info )
                    shapes[item.first] = item.second.shape;
                slog::info << "Reshaping network: " << getShapesString( shapes ) << slog::endl;
                startTime = Time::now();
                cnnNetwork.reshape( shapes );
                auto duration_ms = double_to_string( get_total_ms_time( startTime ) );
                slog::info << "Reshape network took " << duration_ms << " ms" << slog::endl;
                if( statistics )
                    statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                        {
                                {"reshape network time (ms)", duration_ms}
                        } );
            }
            // use batch size according to provided layout and shapes
            batchSize = (!FLAGS_layout.empty()) ? getBatchSize( app_inputs_info ) : cnnNetwork.getBatchSize();

            topology_name = cnnNetwork.getName();
            slog::info << (FLAGS_b != 0 ? "Network batch size was changed to: " : "Network batch size: ") << batchSize << slog::endl;

            // ----------------- 6. Configuring inputs and outputs ----------------------------------------------------------------------
            next_step();

            processPrecision( cnnNetwork, FLAGS_ip, FLAGS_op, FLAGS_iop );
            for( auto& item : cnnNetwork.getInputsInfo() )
            {
                // if precision for input set by user, then set it to app_inputs
                // if it an image, set U8
                if( !FLAGS_ip.empty() || FLAGS_iop.find( item.first ) != std::string::npos )
                {
                    app_inputs_info.at( item.first ).precision = item.second->getPrecision();
                }
                else if( app_inputs_info.at( item.first ).isImage() )
                {
                    app_inputs_info.at( item.first ).precision = Precision::U8;
                    item.second->setPrecision( app_inputs_info.at( item.first ).precision );
                }
            }


            printInputAndOutputsInfo( cnnNetwork );
            // ----------------- 7. Loading the model to the device --------------------------------------------------------
            next_step();
            startTime = Time::now();
            exeNetwork = ie.LoadNetwork( cnnNetwork, device_name );
            duration_ms = double_to_string( get_total_ms_time( startTime ) );
            slog::info << "Load network took " << duration_ms << " ms" << slog::endl;
            if( statistics )
                statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                    {
                            {"load network time (ms)", duration_ms}
                    } );
        }
        else
        {
            next_step();
            slog::info << "Skipping the step for compiled network" << slog::endl;
            next_step();
            slog::info << "Skipping the step for compiled network" << slog::endl;
            next_step();
            slog::info << "Skipping the step for compiled network" << slog::endl;
            // ----------------- 7. Loading the model to the device --------------------------------------------------------
            next_step();
            auto startTime = Time::now();
            exeNetwork = ie.ImportNetwork( FLAGS_m, device_name, {} );
            auto duration_ms = double_to_string( get_total_ms_time( startTime ) );
            slog::info << "Import network took " << duration_ms << " ms" << slog::endl;
            if( statistics )
                statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                    {
                            {"import network time (ms)", duration_ms}
                    } );
            app_inputs_info = getInputsInfo<InputInfo::CPtr>( FLAGS_shape, FLAGS_layout, FLAGS_b, exeNetwork.GetInputsInfo() );
            if( batchSize == 0 )
            {
                batchSize = 1;
            }
        }
        // ----------------- 8. Setting optimal runtime parameters -----------------------------------------------------
        next_step();

        // Update number of streams
        for( auto&& ds : device_nstreams )
        {
            const std::string key = ds.first + "_THROUGHPUT_STREAMS";
            device_nstreams[ds.first] = ie.GetConfig( ds.first, key ).as<std::string>();
        }

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if( nireq == 0 )
        {
            if( FLAGS_api == "sync" )
            {
                nireq = 1;
            }
            else
            {
                std::string key = METRIC_KEY( OPTIMAL_NUMBER_OF_INFER_REQUESTS );
                try
                {
                    nireq = exeNetwork.GetMetric( key ).as<unsigned int>();
                }
                catch( const std::exception& ex )
                {
                    IE_THROW()
                        << "Every device used with the surface_use_case should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << device_name << " with error:" << ex.what();
                }
            }
        }

        // Iteration limit
        uint32_t niter = FLAGS_niter;
        if( (niter > 0) && (FLAGS_api == "async") )
        {
            niter = ((niter + nireq - 1) / nireq) * nireq;
            if( FLAGS_niter != niter )
            {
                slog::warn << "Number of iterations was aligned by request number from "
                    << FLAGS_niter << " to " << niter << " using number of requests " << nireq << slog::endl;
            }
        }

        // Time limit
        uint32_t duration_seconds = 0;
        if( FLAGS_t != 0 )
        {
            // time limit
            duration_seconds = FLAGS_t;
        }
        else if( FLAGS_niter == 0 )
        {
            // default time limit
            duration_seconds = deviceDefaultDeviceDurationInSeconds( device_name );
        }
        uint64_t duration_nanoseconds = getDurationInNanoseconds( duration_seconds );

        if( statistics )
        {
            statistics->addParameters( StatisticsReport::Category::RUNTIME_CONFIG,
                {
                        {"topology", topology_name},
                        {"target device", device_name},
                        {"API", FLAGS_api},
                        {"precision", std::string( precision.name() )},
                        {"batch size", std::to_string( batchSize )},
                        {"number of iterations", std::to_string( niter )},
                        {"number of parallel infer requests", std::to_string( nireq )},
                        {"duration (ms)", std::to_string( getDurationInMilliseconds( duration_seconds ) )},
                } );
            for( auto& nstreams : device_nstreams )
            {
                std::stringstream ss;
                ss << "number of " << nstreams.first << " streams";
                statistics->addParameters( StatisticsReport::Category::RUNTIME_CONFIG,
                    {
                            {ss.str(), nstreams.second},
                    } );
            }
        }

        // ----------------- 9. Creating infer requests and filling input blobs ----------------------------------------
        next_step();

        InferRequestsQueue inferRequestsQueue( exeNetwork, nireq );
        fillBlobs( inputFiles, batchSize, app_inputs_info, inferRequestsQueue.requests );

        // ----------------- 10. Measuring performance ------------------------------------------------------------------
        size_t progressCnt = 0;
        size_t progressBarTotalCount = progressBarDefaultTotalCount;
        size_t iteration = 0;

        std::stringstream ss;
        ss << "Start inference " << FLAGS_api << "hronously";
        if( FLAGS_api == "async" )
        {
            if( !ss.str().empty() )
            {
                ss << ", ";
            }
            ss << nireq << " inference requests";
            std::stringstream device_ss;
            for( auto& nstreams : device_nstreams )
            {
                if( !device_ss.str().empty() )
                {
                    device_ss << ", ";
                }
                device_ss << nstreams.second << " streams for " << nstreams.first;
            }
            if( !device_ss.str().empty() )
            {
                ss << " using " << device_ss.str();
            }
        }
        ss << ", limits: ";
        if( duration_seconds > 0 )
        {
            ss << getDurationInMilliseconds( duration_seconds ) << " ms duration";
        }
        if( niter != 0 )
        {
            if( duration_seconds == 0 )
            {
                progressBarTotalCount = niter;
            }
            if( duration_seconds > 0 )
            {
                ss << ", ";
            }
            ss << niter << " iterations";
        }
        next_step( ss.str() );

        // warming up - out of scope
        auto inferRequest = inferRequestsQueue.getIdleRequest();
        if( !inferRequest )
        {
            IE_THROW() << "No idle Infer Requests!";
        }
        if( FLAGS_api == "sync" )
        {
            inferRequest->infer();
        }
        else
        {
            inferRequest->startAsync();
        }
        inferRequestsQueue.waitAll();
        auto duration_ms = double_to_string( inferRequestsQueue.getLatencies()[0] );
        slog::info << "First inference took " << duration_ms << " ms" << slog::endl;
        if( statistics )
            statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                {
                        {"first inference time (ms)", duration_ms}
                } );
        inferRequestsQueue.resetTimes();

        auto startTime = Time::now();
        auto execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

        /** Start inference & calculate performance **/
        /** to align number if iterations to guarantee that last infer requests are executed in the same conditions **/
        ProgressBar progressBar( progressBarTotalCount, FLAGS_stream_output, FLAGS_progress );

        while( (niter != 0LL && iteration < niter) ||
            (duration_nanoseconds != 0LL && (uint64_t)execTime < duration_nanoseconds) ||
            (FLAGS_api == "async" && iteration % nireq != 0) )
        {
            inferRequest = inferRequestsQueue.getIdleRequest();
            if( !inferRequest )
            {
                IE_THROW() << "No idle Infer Requests!";
            }

            if( FLAGS_api == "sync" )
            {
                inferRequest->infer();
            }
            else
            {
                // As the inference request is currently idle, the wait() adds no additional overhead (and should return immediately).
                // The primary reason for calling the method is exception checking/re-throwing.
                // Callback, that governs the actual execution can handle errors as well,
                // but as it uses just error codes it has no details like ‘what()’ method of `std::exception`
                // So, rechecking for any exceptions here.
                inferRequest->wait();
                inferRequest->startAsync();
            }
            iteration++;

            execTime = std::chrono::duration_cast<ns>(Time::now() - startTime).count();

            if( niter > 0 )
            {
                progressBar.addProgress( 1 );
            }
            else
            {
                // calculate how many progress intervals are covered by current iteration.
                // depends on the current iteration time and time of each progress interval.
                // Previously covered progress intervals must be skipped.
                auto progressIntervalTime = duration_nanoseconds / progressBarTotalCount;
                size_t newProgress = execTime / progressIntervalTime - progressCnt;
                progressBar.addProgress( newProgress );
                progressCnt += newProgress;
            }
        }

        // wait the latest inference executions
        inferRequestsQueue.waitAll();

        double latency = getMedianValue<double>( inferRequestsQueue.getLatencies() );
        double totalDuration = inferRequestsQueue.getDurationInMilliseconds();
        double fps = (FLAGS_api == "sync") ? batchSize * 1000.0 / latency :
            batchSize * 1000.0 * iteration / totalDuration;

        if( statistics )
        {
            statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                {
                        {"total execution time (ms)", double_to_string( totalDuration )},
                        {"total number of iterations", std::to_string( iteration )},
                } );
            if( device_name.find( "MULTI" ) == std::string::npos )
            {
                statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                    {
                            {"latency (ms)", double_to_string( latency )},
                    } );
            }
            statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                {
                        {"throughput", double_to_string( fps )}
                } );
        }

        progressBar.finish();

        // ----------------- 11. Dumping statistics report -------------------------------------------------------------
        next_step();

#ifdef USE_OPENCV
        if( !FLAGS_dump_config.empty() )
        {
            dump_config( FLAGS_dump_config, config );
            slog::info << "Inference Engine configuration settings were dumped to " << FLAGS_dump_config << slog::endl;
        }
#endif

        if( !FLAGS_exec_graph_path.empty() )
        {
            try
            {
                CNNNetwork execGraphInfo = exeNetwork.GetExecGraphInfo();
                execGraphInfo.serialize( FLAGS_exec_graph_path );
                slog::info << "executable graph is stored to " << FLAGS_exec_graph_path << slog::endl;
            }
            catch( const std::exception& ex )
            {
                slog::err << "Can't get executable graph: " << ex.what() << slog::endl;
            }
        }

        if( perf_counts )
        {
            std::vector<std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>> perfCounts;
            for( size_t ireq = 0; ireq < nireq; ireq++ )
            {
                auto reqPerfCounts = inferRequestsQueue.requests[ireq]->getPerformanceCounts();
                if( FLAGS_pc )
                {
                    slog::info << "Performance counts for " << ireq << "-th infer request:" << slog::endl;
                    printPerformanceCounts( reqPerfCounts, std::cout, getFullDeviceName( ie, FLAGS_d ), false );
                }
                perfCounts.push_back( reqPerfCounts );
            }
            if( statistics )
            {
                statistics->dumpPerformanceCounters( perfCounts );
            }
        }

        if( statistics )
            statistics->dump();

        std::cout << "Count:      " << iteration << " iterations" << std::endl;
        std::cout << "Duration:   " << double_to_string( totalDuration ) << " ms" << std::endl;
        if( device_name.find( "MULTI" ) == std::string::npos )
            std::cout << "Latency:    " << double_to_string( latency ) << " ms" << std::endl;
        std::cout << "Throughput: " << double_to_string( fps ) << " FPS" << std::endl;
#endif
    }
    catch( const std::exception& ex )
    {
        slog::err << ex.what() << slog::endl;

        if( statistics )
        {
            statistics->addParameters( StatisticsReport::Category::EXECUTION_RESULTS,
                {
                        {"error", ex.what()},
                } );
            statistics->dump();
        }

        return 3;
    }


    return 0;    
}
