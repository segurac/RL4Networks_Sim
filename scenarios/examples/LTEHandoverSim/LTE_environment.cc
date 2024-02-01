/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *
 */
#include <filesystem>

#include <fstream>

#include <iomanip>

#include <ios>

#include <iostream>

#include <string>

#include <vector>

#include "./mygym.h"

#include <ns3/core-module.h>

#include "ns3/opengym-module.h"

#include <ns3/network-module.h>

#include <ns3/mobility-module.h>

#include <ns3/internet-module.h>

#include <ns3/lte-module.h>

#include <ns3/config-store-module.h>

#include <ns3/point-to-point-helper.h>

#include <ns3/applications-module.h>

#include <ns3/log.h>

#include "ns3/netanim-module.h"

#include "../../src/lte/model/cell-individual-offset.h"


using namespace ns3;

namespace fs = std::filesystem; 

// =========================================================================
// AUX STRUCTS
// =========================================================================
struct random_walk_params {
  int number;
  std::string pos_ini_x_min;
  std::string pos_ini_x_max;
  std::string pos_ini_y_min;
  std::string pos_ini_y_max;
  std::string pos_ini_z_min;
  std::string pos_ini_z_max;
  std::string speed_m_s;
  std::string mov_x_min;
  std::string mov_x_max;
  std::string mov_y_min;
  std::string mov_y_max;
  std::string mov_time_step_sec;
};

// =========================================================================
// AUX FUNCTIONS
// =========================================================================
void NotifyHandoverStartEnb(std::string context,
    uint64_t imsi,
    uint16_t cellid,
    uint16_t rnti,
    uint16_t targetCellId) {
    std::cout << context <<
        " eNB CellId " << cellid <<
        ": start handover of UE with IMSI " << imsi <<
        " RNTI " << rnti <<
        " to CellId " << targetCellId <<
        std::endl;
}

void NotifyHandoverEndOkEnb(std::string context,
    uint64_t imsi,
    uint16_t cellid,
    uint16_t rnti) {
    std::cout << context <<
        " eNB CellId " << cellid <<
        ": completed handover of UE with IMSI " << imsi <<
        " RNTI " << rnti <<
        std::endl;
}

std::vector < double >
    convertStringtoDouble(std::string cioList, uint16_t rep, std::string delimiter) {
        std::vector < double > v;
        double tempVal;
        
        size_t pos = cioList.find(delimiter);
        std::string token;
        while (pos != std::string::npos) {
            token = cioList.substr(0, pos);
            tempVal = std::stod(token);
            for (uint16_t i = 0; i < rep; i++)
                v.push_back(tempVal);

            cioList.erase(0, pos + delimiter.length());
            pos = cioList.find(delimiter);
        }

        tempVal = std::stod(cioList);
        // Every eNB has 3 sectors
        for (uint16_t i = 0; i < rep; i++)
            v.push_back(tempVal);
        //v.push_back(tempVal);
        //v.push_back(tempVal);
        return v;
    }

std::vector <random_walk_params> load_random_walk_params_csv(fs::path csv_path)
{
    std::vector<random_walk_params> output;

    // File pointer 
    std::ifstream file; 
  
    // Open an existing file 
    file.open(csv_path); 
  
    int i = 0;
    int j = 0;
    std::string line, word, temp; 

    // This allows jumping the header line
    std::getline(file, line);

    // read an entire row and 
    // store it in a string variable 'line' 
    while (std::getline(file, line)){
        std::stringstream s(line);
        j = 0;    
        random_walk_params tmp;
        while (std::getline(s, word, ';')) { 
            // add all the column data 
            // of a row to a vector 
            if (j == 0){
                tmp.number = std::stoi(word);
            } else if (j == 1)
            {
                tmp.pos_ini_x_min = word;
            }
            else if (j == 2)
            {
                tmp.pos_ini_x_max = word;
            }
            else if (j == 3)
            {
                tmp.pos_ini_y_min = word;
            }
            else if (j == 4)
            {
                tmp.pos_ini_y_max = word;
            }
            else if (j == 5)
            {
                tmp.pos_ini_z_min = word;
            }
            else if (j == 6)
            {
                tmp.pos_ini_z_max = word;
            }
            else if (j == 7)
            {
                tmp.speed_m_s = word;
            }
            else if (j == 8)
            {
                tmp.mov_x_min = word;
            }
            else if (j == 9)
            {
                tmp.mov_x_max = word;
            }
            else if (j == 10)
            {
                tmp.mov_y_min = word;
            }
            else if (j == 11)
            {
                tmp.mov_y_max = word;
            }
            else if (j == 12)
            {
                tmp.mov_time_step_sec = word;
            }
            j += 1; 
        } 
        output.push_back(tmp);
        i += 1;
    }  
    file.close();

    int count = 0;
    for (auto i = output.begin(); i != output.end(); ++i){
        std::cout << "\t-----------------------------" << std::endl;
        std::cout << "\t\tnumber "<< count << " = " << (*i).number << std::endl;
        std::cout << "\t\tpos_ini_x_min "<< count << " = " << (*i).pos_ini_x_min << std::endl;
        std::cout << "\t\tpos_ini_x_max "<< count << " = " << (*i).pos_ini_x_max << std::endl;
        std::cout << "\t\tpos_ini_y_min "<< count << " = " << (*i).pos_ini_y_min << std::endl;
        std::cout << "\t\tpos_ini_y_max "<< count << " = " << (*i).pos_ini_y_max << std::endl;
        std::cout << "\t\tpos_ini_z_min "<< count << " = " << (*i).pos_ini_z_min << std::endl;
        std::cout << "\t\tpos_ini_z_max "<< count << " = " << (*i).pos_ini_z_max << std::endl;
        std::cout << "\t\tspeed_m_s "<< count << " = " << (*i).speed_m_s << std::endl;
        std::cout << "\t\tmov_x_min "<< count << " = " << (*i).mov_x_min << std::endl;
        std::cout << "\t\tmov_x_max "<< count << " = " << (*i).mov_x_max << std::endl;
        std::cout << "\t\tmov_y_min "<< count << " = " << (*i).mov_y_min << std::endl;
        std::cout << "\t\tmov_y_max "<< count << " = " << (*i).mov_y_max << std::endl;
        std::cout << "\t\tmov_time_step_sec "<< count << " = " << (*i).mov_time_step_sec << std::endl;
        count += 1;
    }

    return output;
}

int** load_adjacency_matrix_csv(fs::path csv_path, int cell_num) 
{ 
  
    // File pointer 
    std::ifstream file; 
  
    // Open an existing file 
    file.open(csv_path); 
  
    // Read the Data from the file 
    // as int Vector 
    int** adjacency_matrix = new int*[cell_num];
    int i = 0;
    int j = 0;
    std::string line, word, temp; 

    // read an entire row and 
    // store it in a string variable 'line' 
    while (std::getline(file, line)){
        adjacency_matrix[i] = new int[cell_num];
        std::stringstream s(line);
        j = 0;      
        while (std::getline(s, word, ';')) { 
            // add all the column data 
            // of a row to a vector 
            adjacency_matrix[i][j] = std::stoi(word);
            j += 1; 
        } 
        i += 1;
    }  
    file.close();

    std::cout << "\tAdjacency Matrix:" << std::endl;
    for (i=0; i<cell_num; i++){
        for (j=0; j<cell_num; j++){
            std::cout << "\t\t" << i << ", " << j << ": " << adjacency_matrix[i][j] << std::endl;
        }
    }
    return adjacency_matrix;
} 

double** load_enb_location_csv(fs::path csv_path, int cell_num) 
{ 
    // File pointer 
    std::ifstream file; 
  
    // Open an existing file 
    file.open(csv_path); 
  
    // Read the Data from the file 
    // as int Vector 
    double** locations = new double*[cell_num];
    int i = 0;
    int j = 0;
    std::string line, word, temp; 

    // read an entire row and 
    // store it in a string variable 'line' 
    while (std::getline(file, line)){
        locations[i] = new double[3];
        std::stringstream s(line);
        j = 0;      
        while (std::getline(s, word, ';')) { 
            // add all the column data 
            // of a row to a vector 
            locations[i][j] = std::stod(word);
            j += 1; 
        } 
        i += 1;
    }  
    file.close();
    return locations;
} 

// =========================================================================
// STATIC METHODS FOR GLOBAL VARIABLES
// =========================================================================
static ns3::GlobalValue g_nMacroEnbSites("nMacroEnbSites",
    "How many macro sites there are",
    ns3::UintegerValue(6),
    ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_eNBAdjacencyMatrixFile("eNBAdjacencyMatrixFile",
    "eNB Adjacency Matrix file (string).",
    ns3::StringValue(""),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_nUEs("nUEs",
    "How many UEs there are",
    ns3::UintegerValue(50),
    ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_ueRandomWalkMobility("ueRandomWalkMobility",
    "File containing the UE Random Walk parameters (string).",
    ns3::StringValue(""),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_ueSimulatedMobility("ueSimulatedMobility",
    "File containing the simulated trajectory of UEs (string).",
    ns3::StringValue(""),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_locationXRange("locationXRange",
    "Min|Max values of X-coordinate.",
    ns3::StringValue("-10000|10000"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_locationYRange("locationYRange",
    "Min|Max values of Y-coordinate.",
    ns3::StringValue("-500|500"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_locationZRange("locationZRange",
    "Min|Max values of Z-coordinate.",
    ns3::StringValue("1|2"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_macroEnbSitesLocationFile("macroEnbSitesLocationFile",
    "File containing the eNB locations (string).",
    ns3::StringValue(""),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_macroEnbTxPowerDbm("macroEnbTxPowerDbm",
    "TX power [dBm] used by macro eNBs",
    ns3::DoubleValue(32.0),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_macroEnbDlEarfcn("macroEnbDlEarfcn",
    "DL EARFCN used by macro eNBs",
    ns3::UintegerValue(100),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_macroEnbUlEarfcnMinusDlEarfcn("macroEnbUlEarfcnMinusDlEarfcn",
    "(UL EARFCN - DL EARFCN) used by macro eNBs",
    ns3::UintegerValue(18000),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_macroEnbBandwidth("macroEnbBandwidth",
    "bandwidth [num RBs] used by macro eNBs",
    ns3::UintegerValue(25),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_macroEnbAntennaModelType("macroEnbAntennaModelType",
    "Antenna model type",
    ns3::StringValue("ns3::IsotropicAntennaModel"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_fadingTrace("fadingTrace",
    "The path of the fading trace (by default no fading trace "
    "is loaded, i.e., fading is not considered)",
    ns3::StringValue(""),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_hystersis("hysteresisCoefficient",
    "The value of hysteresis coefficient",
    ns3::DoubleValue(3.0),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_timeToTrigger("TTT",
    "The value of time to tigger coefficient in milliseconds",
    ns3::DoubleValue(40),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_cioList("cioList",
    "The CIO values arranged in a string.",
    ns3::StringValue("0_0_0_0_0"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_cioRange("cioRange",
    "Range for CIO values arranged in a string.",
    ns3::StringValue("-6.0|6.0"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_simTime("simTime",
    "Total duration of the simulation [s]",
    ns3::DoubleValue(51),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_envStepTime("envStepTime",
    "Environment Step time [s]",
    ns3::DoubleValue(0.2),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_envCollectingWindow("envCollectingWindow",
    "Environment collecting window time [s]",
    ns3::DoubleValue(0.05),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_envRewardType("envRewardType",
    "Environment Reward:"
    "    0: average overall throughput, "
    "    1: PRBs utilization deviation, "
    "    2: number of blocked users",
    ns3::UintegerValue(0),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_userBlockageThr("userBlockageThr",
    "User blockage thresholg [Mb/s]",
    ns3::DoubleValue(0.5),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_openGymPort("openGymPort",
    "Open Gym Port Number",
    ns3::UintegerValue(9999),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_outputTraceFiles("outputTraceFiles",
    "If true, trace files will be output in ns3 working directory. "
    "If false, no files will be generated.",
    ns3::BooleanValue(true),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_epcDl("epcDl",
    "if true, will activate data flows in the downlink when EPC is being used. "
    "If false, downlink flows won't be activated. "
    "If EPC is not used, this parameter will be ignored.",
    ns3::BooleanValue(true),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_epcUl("epcUl",
    "if true, will activate data flows in the uplink when EPC is being used. "
    "If false, uplink flows won't be activated. "
    "If EPC is not used, this parameter will be ignored.",
    ns3::BooleanValue(true),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_useUdp("useUdp",
    "if true, the UdpClient application will be used. "
    "Otherwise, the BulkSend application will be used over a TCP connection. "
    "If EPC is not used, this parameter will be ignored.",
    ns3::BooleanValue(true),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_numBearersPerUe("numBearersPerUe",
    "How many bearers per UE there are in the simulation",
    ns3::UintegerValue(1),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_p2phDataRate("p2phDataRate",
    "Point To Point Data Rate",
    ns3::StringValue("100Mb/s"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_p2phMTU("p2phMTU",
    "Point To Point MTU",
    ns3::UintegerValue(1),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_p2phDelay("p2phDelay",
    "Point To Point Delay (seconds)",
    ns3::DoubleValue(0.010),
    ns3::MakeDoubleChecker<double>());

static ns3::GlobalValue g_lteHelperScheduler("lteHelperScheduler",
    "LTE Helper Scheduler (string).",
    ns3::StringValue("ns3::PfFfMacScheduler"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_lteHelperFfrAlgorithm("lteHelperFfrAlgorithm",
    "LTE Helper FfrAlgorithm (string).",
    ns3::StringValue("ns3::LteFrNoOpAlgorithm"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_lteHelperHandoverAlgorithm("lteHelperHandoverAlgorithm",
    "LTE Helper Handover Algorithm (string).",
    ns3::StringValue("ns3::NoOpHandoverAlgorithm"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_lteHelperPathlossModel("lteHelperPathlossModel",
    "LTE Helper Path loss Model (string).",
    ns3::StringValue("ns3::FriisPropagationLossModel"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_lteHelperFadingModel("lteHelperFadingModel",
    "LTE Helper Fading Model (string).",
    ns3::StringValue(""),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_lteHelperUseIdealRrc("lteHelperUseIdealRrc",
    "LTE Helper Use Ideal Rrc.",
    ns3::BooleanValue(true),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_lteHelperAnrEnabled("lteHelperAnrEnabled",
    "LTE Helper Anr Enabled.",
    ns3::BooleanValue(true),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_lteHelperEnbComponentCarrierManager("lteHelperEnbComponentCarrierManager",
    "LTE Helper EnbComponentCarrierManagerl (string).",
    ns3::StringValue("ns3::NoOpComponentCarrierManager"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_lteHelperUeComponentCarrierManager("lteHelperUeComponentCarrierManager",
    "LTE Helper UeComponentCarrierManager (string).",
    ns3::StringValue("ns3::SimpleUeComponentCarrierManager"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_llteHelperUseCa("lteHelperUseCa",
    "LTE Helper Use Ca.",
    ns3::BooleanValue(false),
    ns3::MakeBooleanChecker());

static ns3::GlobalValue g_lteHelperNumberOfComponentCarriers("lteHelperNumberOfComponentCarriers",
    "LTE Helper NumberOfComponentCarriers",
    ns3::UintegerValue(1),
    ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_lteHelperSpectrumChannelType("lteHelperSpectrumChannelType",
    "LTE Helper SpectrumChannelType (string).",
    ns3::StringValue("ns3::MultiModelSpectrumChannel"),
    ns3::MakeStringChecker());

static ns3::GlobalValue g_srsPeriodicity("srsPeriodicity",
    "SRS Periodicity (has to be at least "
    "greater than the number of UEs per eNB)",
    ns3::UintegerValue(80),
    ns3::MakeUintegerChecker<uint16_t>());

static ns3::GlobalValue g_updClientInterval("updClientInterval",
    "Milliseconds (uint_32)",
    ns3::UintegerValue(10),
    ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_udpClientPacketSize("udpClientPacketSize",
    "bytes (uint_32)",
    ns3::UintegerValue(1200),
    ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_udpClientMaxPacketse("udpClientMaxPackets",
    "(uint_32)",
    ns3::UintegerValue(10000000),
    ns3::MakeUintegerChecker<uint32_t>());

static ns3::GlobalValue g_lteRlcUmMaxTxBufferSize("lteRlcUmMaxTxBufferSize",
    "(uint_32)",
    ns3::UintegerValue(10240),
    ns3::MakeUintegerChecker<uint32_t>());

// =========================================================================
// MAIN SCRIPT
// =========================================================================
uint32_t RunNum;
std::string ConfigFile;

int main(int argc, char * argv[]) {
    std::cout << "======================================================" << std::endl;
    std::cout << "               START TOY ENVIRONMENT                  " << std::endl;
    std::cout << "======================================================" << std::endl;

    CommandLine cmd;
    cmd.AddValue("RunNum", "1...10", RunNum);
    cmd.AddValue("ConfigFile", "path", ConfigFile);
    cmd.Parse(argc, argv);

    fs::path ns3_path = fs::current_path();
    fs::path full_path = ns3_path / ConfigFile;

    std::cout << "Current path: " << fs::current_path() << std::endl;
    std::cout << "Config path: " << full_path << std::endl;

    Config::SetDefault ("ns3::ConfigStore::Filename", StringValue (full_path));
    Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Load"));
    Config::SetDefault ("ns3::ConfigStore::FileFormat", StringValue ("Xml"));
    
    ConfigStore inputConfig;
    inputConfig.ConfigureDefaults();

    if (RunNum < 1) RunNum = 1;
    SeedManager::SetSeed(1);
    SeedManager::SetRun(RunNum);

    // **************************************
    // Recover global values
    // **************************************
    std::cout << "Environment parameters: " << std::endl;
    UintegerValue uintegerValue;
    IntegerValue integerValue;
    DoubleValue doubleValue;
    BooleanValue booleanValue;
    StringValue stringValue;
    std::string tmp1;
    fs::path tmp2;

    GlobalValue::GetValueByName("locationXRange", stringValue);
    std::string locationXRange = stringValue.Get();
    std::cout << "\tlocationXRange: " << locationXRange << std::endl;

    GlobalValue::GetValueByName("locationYRange", stringValue);
    std::string locationYRange = stringValue.Get();
    std::cout << "\tlocationYRange: " << locationYRange << std::endl;

    GlobalValue::GetValueByName("locationZRange", stringValue);
    std::string locationZRange = stringValue.Get();
    std::cout << "\tlocationZRange: " << locationZRange << std::endl;

    GlobalValue::GetValueByName("nUEs", uintegerValue);
    uint32_t nUEs = uintegerValue.Get();
    std::cout << "\tnUEs: " << nUEs << std::endl;

    GlobalValue::GetValueByName("ueRandomWalkMobility", stringValue);
    tmp1 = stringValue.Get();
    tmp2 = tmp1;
    fs::path ueRandomWalkMobility;
    if (tmp2.empty()){
        ueRandomWalkMobility = tmp2;
    } else {
        ueRandomWalkMobility = ns3_path / tmp2;
    };
    std::cout << "\tueRandomWalkMobility: " << ueRandomWalkMobility << std::endl;
    std::vector <random_walk_params> randomWalkInfo;
    if (!ueRandomWalkMobility.empty()){
        randomWalkInfo = load_random_walk_params_csv(ueRandomWalkMobility);
    }
    GlobalValue::GetValueByName("ueSimulatedMobility", stringValue);
    tmp1 = stringValue.Get();
    tmp2 = tmp1;
    fs::path ueSimulatedMobility;
    if (tmp2.empty()){
        ueSimulatedMobility = tmp2;
    } else {
        ueSimulatedMobility = ns3_path / tmp2;
    };
    std::cout << "\tueSimulatedMobility: " << ueSimulatedMobility << std::endl;

    GlobalValue::GetValueByName("nMacroEnbSites", uintegerValue);
    uint32_t nMacroEnbSites = uintegerValue.Get();
    std::cout << "\tnMacroEnbSites: " << nMacroEnbSites << std::endl;

    GlobalValue::GetValueByName("macroEnbSitesLocationFile", stringValue);
    tmp1 = stringValue.Get();
    tmp2 = tmp1;
    fs::path macroEnbSitesLocationFile = ns3_path / tmp2;
    std::cout << "\tmacroEnbSitesLocationFile: " << macroEnbSitesLocationFile << std::endl;
    double** macroEnbLocations = load_enb_location_csv(macroEnbSitesLocationFile, nMacroEnbSites);

    GlobalValue::GetValueByName("eNBAdjacencyMatrixFile", stringValue);
    tmp1 = stringValue.Get();
    tmp2 = tmp1;
    fs::path eNBAdjacencyMatrixFile = ns3_path / tmp2;
    std::cout << "\teNBAdjacencyMatrixFile: " << eNBAdjacencyMatrixFile << std::endl;
    int** adjacency_matrix = load_adjacency_matrix_csv(eNBAdjacencyMatrixFile, nMacroEnbSites);
    int nEdgeNumber = 0;
    for(int i1 = 0; i1 < int(nMacroEnbSites) ; i1++){
        for (int i2 = 0; i2 < i1; i2++){
            nEdgeNumber += adjacency_matrix[i1][i2];
        }
    }
    std::cout << "\tNumber of Edges (ones in Adj. Matrix): " << nEdgeNumber << std::endl;

    GlobalValue::GetValueByName("macroEnbTxPowerDbm", doubleValue);
    double macroEnbTxPowerDbm = doubleValue.Get();
    std::cout << "\tmacroEnbTxPowerDbm: " << macroEnbTxPowerDbm << std::endl;
    
    GlobalValue::GetValueByName("macroEnbDlEarfcn", uintegerValue);
    uint32_t macroEnbDlEarfcn = uintegerValue.Get();
    std::cout << "\tmacroEnbDlEarfcn: " << macroEnbDlEarfcn << std::endl;

    GlobalValue::GetValueByName("macroEnbUlEarfcnMinusDlEarfcn", uintegerValue);
    uint32_t macroEnbUlEarfcnMinusDlEarfcn = uintegerValue.Get();
    std::cout << "\tmacroEnbUlEarfcnMinusDlEarfcn: " << macroEnbUlEarfcnMinusDlEarfcn << std::endl;
    
    GlobalValue::GetValueByName("macroEnbBandwidth", uintegerValue);
    uint16_t macroEnbBandwidth = uintegerValue.Get();
    std::cout << "\tmacroEnbBandwidth: " << macroEnbBandwidth << std::endl;

    GlobalValue::GetValueByName("macroEnbAntennaModelType", stringValue);
    std::string macroEnbAntennaModelType = stringValue.Get();
    std::cout << "\tmacroEnbAntennaModelType: " << macroEnbAntennaModelType << std::endl;

    GlobalValue::GetValueByName("fadingTrace", stringValue);
    std::string fadingTrace = stringValue.Get();
    std::cout << "\tfadingTrace: " << fadingTrace << std::endl;
    
    GlobalValue::GetValueByName("hysteresisCoefficient", doubleValue);
    double hysteresisCoefficient = doubleValue.Get();
    std::cout << "\thysteresisCoefficient: " << hysteresisCoefficient << std::endl;
    
    GlobalValue::GetValueByName("TTT", doubleValue);
    double timeToTrigger = doubleValue.Get();
    std::cout << "\ttimeToTrigger: " << timeToTrigger << std::endl;

    GlobalValue::GetValueByName("cioList", stringValue);
    std::string cioList = stringValue.Get();
    std::cout << "\tcioList: " << cioList << std::endl;

    GlobalValue::GetValueByName("cioRange", stringValue);
    std::string cioRange = stringValue.Get();
    std::cout << "\tcioRange: " << cioRange << std::endl;
    
    GlobalValue::GetValueByName("simTime", doubleValue);
    double simTime = doubleValue.Get();
    std::cout << "\tsimTime: " << simTime << std::endl;
    
    GlobalValue::GetValueByName("envStepTime", doubleValue);
    double envStepTime = doubleValue.Get();
    std::cout << "\tenvStepTime: " << envStepTime << std::endl;

    GlobalValue::GetValueByName("envCollectingWindow", doubleValue);
    double envCollectingWindow = doubleValue.Get();
    std::cout << "\tenvCollectingWindow: " << envCollectingWindow << std::endl;

    GlobalValue::GetValueByName("envRewardType", uintegerValue);
    uint16_t envRewardType = uintegerValue.Get();
    std::cout << "\tenvRewardType: " << envRewardType << std::endl;

    GlobalValue::GetValueByName("userBlockageThr", doubleValue);
    double userBlockageThr = doubleValue.Get();
    std::cout << "\tuserBlockageThr: " << userBlockageThr << std::endl;
    
    GlobalValue::GetValueByName("openGymPort", uintegerValue);
    uint16_t openGymPort = uintegerValue.Get();
    std::cout << "\topenGymPort: " << openGymPort << std::endl;
    
    GlobalValue::GetValueByName("outputTraceFiles", booleanValue);
    bool outputTraceFiles = booleanValue.Get();
    std::cout << "\toutputTraceFiles: " << outputTraceFiles << std::endl;
    
    GlobalValue::GetValueByName("epcDl", booleanValue);
    bool epcDl = booleanValue.Get();
    std::cout << "\tepcDl: " << epcDl << std::endl;
    
    GlobalValue::GetValueByName("epcUl", booleanValue);
    bool epcUl = booleanValue.Get();
    std::cout << "\tepcUl: " << epcUl << std::endl;
    
    GlobalValue::GetValueByName("useUdp", booleanValue);
    bool useUdp = booleanValue.Get();
    std::cout << "\tuseUdp: " << useUdp << std::endl;
    
    GlobalValue::GetValueByName("numBearersPerUe", uintegerValue);
    uint16_t numBearersPerUe = uintegerValue.Get();
    std::cout << "\tnumBearersPerUe: " << numBearersPerUe << std::endl;

    GlobalValue::GetValueByName("p2phDataRate", stringValue);
    std::string p2phDataRate = stringValue.Get();
    std::cout << "\tp2phDataRate: " << p2phDataRate << std::endl;

    GlobalValue::GetValueByName("p2phMTU", uintegerValue);
    uint16_t p2phMTU = uintegerValue.Get();
    std::cout << "\tp2phMTU: " << p2phMTU << std::endl;

    GlobalValue::GetValueByName("p2phDelay", doubleValue);
    double p2phDelay = doubleValue.Get();
    std::cout << "\tp2phDelay: " << p2phDelay << std::endl;

    GlobalValue::GetValueByName("lteHelperScheduler", stringValue);
    std::string lteHelperScheduler = stringValue.Get();
    std::cout << "\tns3::LteHelper::Scheduler: " << lteHelperScheduler << std::endl;

    GlobalValue::GetValueByName("lteHelperFfrAlgorithm", stringValue);
    std::string lteHelperFfrAlgorithm = stringValue.Get();
    std::cout << "\tns3::LteHelper::FfrAlgorithm: " << lteHelperFfrAlgorithm << std::endl;
    
    GlobalValue::GetValueByName("lteHelperHandoverAlgorithm", stringValue);
    std::string lteHelperHandoverAlgorithm = stringValue.Get();
    std::cout << "\tns3::LteHelper::HandoverAlgorithm: " << lteHelperHandoverAlgorithm << std::endl;
    
    GlobalValue::GetValueByName("lteHelperPathlossModel", stringValue);
    std::string lteHelperPathlossModel = stringValue.Get();
    std::cout << "\tns3::LteHelper::PathlossModel: " << lteHelperPathlossModel << std::endl;

    GlobalValue::GetValueByName("lteHelperFadingModel", stringValue);
    std::string lteHelperFadingModel = stringValue.Get();
    std::cout << "\tns3::LteHelper::FadingModel: " << lteHelperFadingModel << std::endl;

    GlobalValue::GetValueByName("lteHelperUseIdealRrc", booleanValue);
    bool lteHelperUseIdealRrc = booleanValue.Get();
    std::cout << "\tns3::LteHelper::UseIdealRrc: " << lteHelperUseIdealRrc << std::endl;
    
    GlobalValue::GetValueByName("lteHelperAnrEnabled", booleanValue);
    bool lteHelperAnrEnabled = booleanValue.Get();
    std::cout << "\tns3::LteHelper::AnrEnabled: " << lteHelperAnrEnabled << std::endl;
    
    GlobalValue::GetValueByName("lteHelperEnbComponentCarrierManager", stringValue);
    std::string lteHelperEnbComponentCarrierManager = stringValue.Get();
    std::cout << "\tns3::LteHelper::EnbComponentCarrierManager: " << lteHelperEnbComponentCarrierManager << std::endl;

    GlobalValue::GetValueByName("lteHelperUeComponentCarrierManager", stringValue);
    std::string lteHelperUeComponentCarrierManager = stringValue.Get();
    std::cout << "\tns3::LteHelper::UeComponentCarrierManager: " << lteHelperUeComponentCarrierManager << std::endl;
    
    GlobalValue::GetValueByName("lteHelperUseCa", booleanValue);
    bool lteHelperUseCa = booleanValue.Get();
    std::cout << "\tns3::LteHelper::UseCa: " << lteHelperUseCa << std::endl;

    GlobalValue::GetValueByName("lteHelperNumberOfComponentCarriers", uintegerValue);
    uint32_t lteHelperNumberOfComponentCarriers = uintegerValue.Get();
    std::cout << "\tns3::LteHelper::NumberOfComponentCarriers: " << lteHelperNumberOfComponentCarriers << std::endl;

    GlobalValue::GetValueByName("lteHelperSpectrumChannelType", stringValue);
    std::string lteHelperSpectrumChannelType = stringValue.Get();
    std::cout << "\tns3::LteHelper::SpectrumChannelType: " << lteHelperSpectrumChannelType << std::endl;
    
    GlobalValue::GetValueByName("srsPeriodicity", uintegerValue);
    uint16_t srsPeriodicity = uintegerValue.Get();
    std::cout << "\tns3::LteEnbRrc::SrsPeriodicity: " << srsPeriodicity << std::endl;

    GlobalValue::GetValueByName("updClientInterval", uintegerValue);
    std::int64_t updClientInterval = uintegerValue.Get();
    std::cout << "\tns3::UdpClient::Interval: " << updClientInterval << std::endl;

    GlobalValue::GetValueByName("udpClientPacketSize", uintegerValue);
    std::int32_t udpClientPacketSize = uintegerValue.Get();
    std::cout << "\tns3::UdpClient::PacketSize: " << udpClientPacketSize << std::endl;

    GlobalValue::GetValueByName("udpClientMaxPackets", uintegerValue);
    std::int32_t udpClientMaxPackets = uintegerValue.Get();
    std::cout << "\tns3::UdpClient::MaxPackets: " << udpClientMaxPackets << std::endl;

    GlobalValue::GetValueByName("lteRlcUmMaxTxBufferSize", uintegerValue);
    std::int32_t lteRlcUmMaxTxBufferSize = uintegerValue.Get();
    std::cout << "\tns3::LteRlcUm::MaxTxBufferSize: " << lteRlcUmMaxTxBufferSize << std::endl;

    // **************************************
    // Setting default values
    // **************************************
    Config::SetDefault("ns3::LteHelper::Scheduler", StringValue(lteHelperScheduler));
    Config::SetDefault("ns3::LteHelper::FfrAlgorithm", StringValue(lteHelperFfrAlgorithm));
    Config::SetDefault("ns3::LteHelper::HandoverAlgorithm", StringValue(lteHelperHandoverAlgorithm));
    Config::SetDefault("ns3::LteHelper::PathlossModel", StringValue(lteHelperPathlossModel));
    Config::SetDefault("ns3::LteHelper::FadingModel", StringValue(lteHelperFadingModel));
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(lteHelperUseIdealRrc));
    Config::SetDefault("ns3::LteHelper::AnrEnabled", BooleanValue(lteHelperAnrEnabled));
    Config::SetDefault("ns3::LteHelper::EnbComponentCarrierManager", StringValue(lteHelperEnbComponentCarrierManager));
    Config::SetDefault("ns3::LteHelper::UeComponentCarrierManager", StringValue(lteHelperUeComponentCarrierManager));
    Config::SetDefault("ns3::LteHelper::UseCa", BooleanValue(lteHelperUseCa));
    Config::SetDefault("ns3::LteHelper::NumberOfComponentCarriers", UintegerValue(lteHelperNumberOfComponentCarriers));

    Config::SetDefault("ns3::UdpClient::Interval", TimeValue(MilliSeconds(updClientInterval)));
    Config::SetDefault("ns3::UdpClient::PacketSize", UintegerValue(udpClientPacketSize));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(udpClientMaxPackets));
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(lteRlcUmMaxTxBufferSize));
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(srsPeriodicity));
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(macroEnbTxPowerDbm));

    // **************************************
    // Create GYM env
    // **************************************
    std::cout << "Creating GYM env..." << std::endl;
    std::string delimiter = "|";
    std::vector < double > cioRangeDouble = convertStringtoDouble(cioRange, 1, delimiter);
    Ptr < OpenGymInterface > openGymInterface = CreateObject < OpenGymInterface > (openGymPort);
    Ptr < MyGymEnv > myGymEnv = CreateObject < MyGymEnv > (envStepTime, nMacroEnbSites, nUEs, macroEnbBandwidth, envCollectingWindow, 
                                                           userBlockageThr, envRewardType, nEdgeNumber, cioRangeDouble);
    myGymEnv -> SetOpenGymInterface(openGymInterface);

    // **************************************
    // Create LTE Helper
    // **************************************
    std::cout << "Creating LTE Helper..." << std::endl;
    Ptr < LteHelper > lteHelper = CreateObject < LteHelper > ();
    lteHelper -> SetAttribute("PathlossModel", StringValue(lteHelperPathlossModel));
    lteHelper -> SetSpectrumChannelType(lteHelperSpectrumChannelType);

    // Set handover parameters
    std::cout << "\tHandover params..." << std::endl;
    lteHelper -> SetHandoverAlgorithmType(lteHelperHandoverAlgorithm);
    lteHelper -> SetHandoverAlgorithmAttribute("Hysteresis", DoubleValue(hysteresisCoefficient));
    lteHelper -> SetHandoverAlgorithmAttribute("TimeToTrigger", TimeValue(MilliSeconds(timeToTrigger)));

    // TODO: 
    // 1) Adjacency matrix to config file  --> OK
    // 2) Add Adjency matrix to CellIndividualOffeset --> OK
    // 3) Add hysteresis to CellIndividualOffeset --> TODO

    CellIndividualOffset::setCellNum(nMacroEnbSites);
    CellIndividualOffset::setAdjacencyMatrix(adjacency_matrix);
    delimiter = "_";
    std::vector < double > cioListDouble = convertStringtoDouble(cioList, 1, delimiter);
    CellIndividualOffset::setOffsetList(cioListDouble);
    std::cout << "\t\tHysteresis = " <<  hysteresisCoefficient << "dB" << std::endl;
    std::cout << "\t\tTimeToTrigger = " <<  timeToTrigger << "ms" << std::endl;
    std::int16_t count;
    count = 1;
    for (auto i = cioListDouble.begin(); i != cioListDouble.end(); ++i){
        std::cout << "\t\tcio "<< count << " = " << *i << std::endl;
        count += 1;
    }
        
    // Set fading model
    NS_LOG_LOGIC(" ====== Setting up fading ====== ");
    if (!fadingTrace.empty()) {
        std::cout << "\tFading model..." << std::endl;
        lteHelper -> SetAttribute("FadingModel", StringValue(lteHelperFadingModel));
        lteHelper -> SetFadingModelAttribute("TraceFilename", StringValue(fadingTrace));
        std::cout << "\t\tFilename " << fadingTrace << std::endl;
    } else {
        std::cout << "\tNo fading specified..." << std::endl;
    }

    // Set EPC helper
    Ptr < PointToPointEpcHelper > epcHelper = CreateObject < PointToPointEpcHelper > ();
    lteHelper -> SetEpcHelper(epcHelper);

    // Set other scenrio paratmeters
    lteHelper -> SetEnbAntennaModelType(macroEnbAntennaModelType);
    lteHelper -> SetEnbDeviceAttribute("DlEarfcn", UintegerValue(macroEnbDlEarfcn));
    lteHelper -> SetEnbDeviceAttribute("UlEarfcn", UintegerValue(macroEnbDlEarfcn + macroEnbUlEarfcnMinusDlEarfcn));
    lteHelper -> SetEnbDeviceAttribute("DlBandwidth", UintegerValue(macroEnbBandwidth));
    lteHelper -> SetEnbDeviceAttribute("UlBandwidth", UintegerValue(macroEnbBandwidth));


    // **************************************
    // eNB deployment
    // **************************************
    std::cout << "Deploying eNBs..." << std::endl;
    NodeContainer macroEnbs;
    macroEnbs.Create(nMacroEnbSites);

    Ptr < ListPositionAllocator > enbPositionAlloc = CreateObject < ListPositionAllocator > ();

    for (unsigned i = 0; i < nMacroEnbSites; ++i){
        enbPositionAlloc->Add(Vector(macroEnbLocations[i][0], 
                                     macroEnbLocations[i][1], 
                                     macroEnbLocations[i][2]));
    }

    MobilityHelper mobility;
    // NOTE: eNB Mobility is not a parameter, they are static by definition
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.SetPositionAllocator(enbPositionAlloc);
    mobility.Install(macroEnbs);
    NetDeviceContainer macroEnbDevs = lteHelper -> InstallEnbDevice(macroEnbs);

    std::vector < Vector > eNBsLocation;
    Vector tempLocation;
    for (uint32_t it = 0; it != macroEnbs.GetN(); ++it) {
        Ptr < Node > node = macroEnbs.Get(it);
        Ptr < NetDevice > netDevice = macroEnbDevs.Get(it);
        Ptr < LteEnbNetDevice > enbNetDevice = netDevice -> GetObject < LteEnbNetDevice > ();
        Ptr < MobilityModel > mobilityModel = node -> GetObject < MobilityModel > ();
        tempLocation = mobilityModel -> GetPosition();
        eNBsLocation.push_back(tempLocation);
        std::cout << "\tCell #" << enbNetDevice -> GetCellId() << " Pos =  " << tempLocation << ", CIO = " << cioListDouble[it] << std::endl;
    }

    // this enables handover for macro eNBs
    lteHelper -> AddX2Interface(macroEnbs);

    // **************************************
    // UE deployment
    // **************************************
    std::cout << "Deploying UEs..." << std::endl;
    
    // Install Mobility Model in UE
    std::vector < NodeContainer > VecNodeCointaner;
    std::vector < NetDeviceContainer > VecNetDeviceContainer;
    if (ueSimulatedMobility.empty()){
        std::cout << "Random walk mobility..." << std::endl;
        for (auto i = randomWalkInfo.begin(); i != randomWalkInfo.end(); ++i){
            std::cout << "\tnumber: " << (*i).number << std::endl;
            NodeContainer tmpUes;
            tmpUes.Create((*i).number);

            MobilityHelper UesMobility;
            int64_t m_streamIndex = 0;
            ObjectFactory pos;
            pos.SetTypeId("ns3::RandomBoxPositionAllocator");
            pos.Set("X", StringValue("ns3::UniformRandomVariable[Min=" + (*i).pos_ini_x_min + "|Max=" + (*i).pos_ini_x_max + "]"));
            pos.Set("Y", StringValue("ns3::UniformRandomVariable[Min=" + (*i).pos_ini_y_min + "|Max=" + (*i).pos_ini_y_max + "]"));
            pos.Set("Z", StringValue("ns3::UniformRandomVariable[Min=" + (*i).pos_ini_z_min + "|Max=" + (*i).pos_ini_z_max + "]"));

            Ptr < PositionAllocator > taPositionAlloc = pos.Create() -> GetObject < PositionAllocator > ();
            m_streamIndex += taPositionAlloc -> AssignStreams(m_streamIndex);

            UesMobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                "Mode", StringValue("Time"),
                "Time", StringValue((*i).mov_time_step_sec),
                "Speed", StringValue("ns3::ConstantRandomVariable[Constant=" + (*i).speed_m_s+ "]"),
                "Bounds", StringValue((*i).mov_x_min + "|" + (*i).mov_x_max + "|" + (*i).mov_y_min + "|" + (*i).mov_y_max));
            UesMobility.SetPositionAllocator(taPositionAlloc);
            UesMobility.Install(tmpUes);
            m_streamIndex += UesMobility.AssignStreams(tmpUes, m_streamIndex);
            NetDeviceContainer UeDevs = lteHelper -> InstallUeDevice(tmpUes);

            for (uint32_t it = 0; it != tmpUes.GetN(); ++it) {
                Ptr < Node > node = tmpUes.Get(it);
                Ptr < NetDevice > netDevice = UeDevs.Get(it);
                Ptr < LteUeNetDevice > uebNetDevice = netDevice -> GetObject < LteUeNetDevice > ();
                Ptr < MobilityModel > mobilityModel = node -> GetObject < MobilityModel > ();
                tempLocation = mobilityModel -> GetPosition();
                eNBsLocation.push_back(tempLocation);
                std::cout <<"\t\tUE #" << uebNetDevice -> GetImsi() << " Initial Pos =  " << tempLocation << std::endl;
            }
            VecNodeCointaner.push_back(tmpUes);
            VecNetDeviceContainer.push_back(UeDevs);
        }
    } else {
        std::cout << "Simulated mobility..." << std::endl;
        Ns2MobilityHelper ns2 = Ns2MobilityHelper(ueSimulatedMobility);
        NodeContainer tmpUes;
        tmpUes.Create(nUEs);
        ns2.Install(tmpUes.Begin(), tmpUes.End()); // configure movements for each node, while reading trace file
        NetDeviceContainer UeDevs = lteHelper->InstallUeDevice(tmpUes);
        for (uint32_t it = 0; it != tmpUes.GetN(); ++it) {
                Ptr < Node > node = tmpUes.Get(it);
                Ptr < NetDevice > netDevice = UeDevs.Get(it);
                Ptr < LteUeNetDevice > uebNetDevice = netDevice -> GetObject < LteUeNetDevice > ();
                Ptr < MobilityModel > mobilityModel = node -> GetObject < MobilityModel > ();
                tempLocation = mobilityModel -> GetPosition();
                eNBsLocation.push_back(tempLocation);
                std::cout <<"\t\tUE #" << uebNetDevice -> GetImsi() << " Initial Pos =  " << tempLocation << std::endl;
            }
        VecNodeCointaner.push_back(tmpUes);
        VecNetDeviceContainer.push_back(UeDevs);

        // NOTE: Code below allows to generate a file with UEs movement trace
        // AsciiTraceHelper ascii;
        // MobilityHelper::EnableAsciiAll (ascii.CreateFileStream ("mobility-trace-example_5.mob"));
    }

    // **************************************
    // Internet and Remote Host
    // **************************************
    std::cout << "Setting up internet and create a single remote host..." << std::endl;
    
    // Create a single RemoteHost
    Ptr < Node > remoteHost;
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    remoteHost = remoteHostContainer.Get(0);
    InternetStackHelper internet;
    internet.Install(remoteHostContainer);
    Ipv4Address remoteHostAddr;

    NodeContainer ues;
    Ipv4InterfaceContainer ueIpIfaces;
    NetDeviceContainer ueDevs;

    // Create the Internet
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate(p2phDataRate)));   
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(p2phMTU));  
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(p2phDelay)));  
    Ptr < Node > pgw = epcHelper -> GetPgwNode();
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    // in this container, interface 0 is the pgw, 1 is the remoteHost
    remoteHostAddr = internetIpIfaces.GetAddress(1);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr < Ipv4StaticRouting > remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting(remoteHost -> GetObject < Ipv4 > ());
    remoteHostStaticRouting -> AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // for internetworking purposes, consider together Edge UEs and macro UEs
    for (auto i = VecNodeCointaner.begin(); i != VecNodeCointaner.end(); ++i){
        ues.Add(*i);
    }
    for (auto i = VecNetDeviceContainer.begin(); i != VecNetDeviceContainer.end(); ++i){
        ueDevs.Add(*i);
    }

    // Install the IP stack on the UEs
    internet.Install(ues);
    ueIpIfaces = epcHelper -> AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    // attachment (needs to be done after IP stack configuration)
    // using initial cell selection
    for (auto i = VecNetDeviceContainer.begin(); i != VecNetDeviceContainer.end(); ++i){
        lteHelper -> Attach(*i);
    }

    // **************************************
    // Applications
    // **************************************
    std::cout << "Setting up applications..." << std::endl;

    // Install and start applications on UEs and remote host
    uint16_t dlPort = 10000;
    uint16_t ulPort = 20000;

    // randomize a bit start times to avoid simulation artifacts
    // (e.g., buffer overflows due to packet transmissions happening
    // exactly at the same time)
    Ptr < UniformRandomVariable > startTimeSeconds = CreateObject < UniformRandomVariable > ();
    if (useUdp) {
        startTimeSeconds -> SetAttribute("Min", DoubleValue(0));
        startTimeSeconds -> SetAttribute("Max", DoubleValue(0.05));
    } else {
        // TCP needs to be started late enough so that all UEs are connected
        // otherwise TCP SYN packets will get lost
        startTimeSeconds -> SetAttribute("Min", DoubleValue(0.100));
        startTimeSeconds -> SetAttribute("Max", DoubleValue(0.110));
    }

    for (uint32_t u = 0; u < ues.GetN(); ++u) {
        Ptr < Node > ue = ues.Get(u);
        // Set the default gateway for the UE
        Ptr < Ipv4StaticRouting > ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(ue -> GetObject < Ipv4 > ());
        ueStaticRouting -> SetDefaultRoute(epcHelper -> GetUeDefaultGatewayAddress(), 1);

        for (uint32_t b = 0; b < numBearersPerUe; ++b) {
            ++dlPort;
            ++ulPort;

            ApplicationContainer clientApps;
            ApplicationContainer serverApps;

            if (useUdp) {
                if (epcDl) {
                    NS_LOG_LOGIC("installing UDP DL app for UE " << u);
                    UdpClientHelper dlClientHelper(ueIpIfaces.GetAddress(u), dlPort);
                    clientApps.Add(dlClientHelper.Install(remoteHost));
                    PacketSinkHelper dlPacketSinkHelper("ns3::UdpSocketFactory",
                        InetSocketAddress(Ipv4Address::GetAny(), dlPort));
                    serverApps.Add(dlPacketSinkHelper.Install(ue));
                }
                if (epcUl) {
                    NS_LOG_LOGIC("installing UDP UL app for UE " << u);
                    UdpClientHelper ulClientHelper(remoteHostAddr, ulPort);
                    clientApps.Add(ulClientHelper.Install(ue));
                    PacketSinkHelper ulPacketSinkHelper("ns3::UdpSocketFactory",
                        InetSocketAddress(Ipv4Address::GetAny(), ulPort));
                    serverApps.Add(ulPacketSinkHelper.Install(remoteHost));
                }
            } else // use TCP
            {
                if (epcDl) {
                    NS_LOG_LOGIC("installing TCP DL app for UE " << u);
                    BulkSendHelper dlClientHelper("ns3::TcpSocketFactory",
                        InetSocketAddress(ueIpIfaces.GetAddress(u), dlPort));
                    dlClientHelper.SetAttribute("MaxBytes", UintegerValue(0));
                    clientApps.Add(dlClientHelper.Install(remoteHost));
                    PacketSinkHelper dlPacketSinkHelper("ns3::TcpSocketFactory",
                        InetSocketAddress(Ipv4Address::GetAny(), dlPort));
                    serverApps.Add(dlPacketSinkHelper.Install(ue));
                }
                if (epcUl) {
                    NS_LOG_LOGIC("installing TCP UL app for UE " << u);
                    BulkSendHelper ulClientHelper("ns3::TcpSocketFactory",
                        InetSocketAddress(remoteHostAddr, ulPort));
                    ulClientHelper.SetAttribute("MaxBytes", UintegerValue(0));
                    clientApps.Add(ulClientHelper.Install(ue));
                    PacketSinkHelper ulPacketSinkHelper("ns3::TcpSocketFactory",
                        InetSocketAddress(Ipv4Address::GetAny(), ulPort));
                    serverApps.Add(ulPacketSinkHelper.Install(remoteHost));
                }
            } // end if (useUdp)

            Ptr < EpcTft > tft = Create < EpcTft > ();
            if (epcDl) {
                EpcTft::PacketFilter dlpf;
                dlpf.localPortStart = dlPort;
                dlpf.localPortEnd = dlPort;
                tft -> Add(dlpf);
            }
            if (epcUl) {
                EpcTft::PacketFilter ulpf;
                ulpf.remotePortStart = ulPort;
                ulpf.remotePortEnd = ulPort;
                tft -> Add(ulpf);
            }

            if (epcDl || epcUl) {
                EpsBearer bearer(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
                lteHelper -> ActivateDedicatedEpsBearer(ueDevs.Get(u), bearer, tft);
            }
            Time startTime = Seconds(startTimeSeconds -> GetValue());
            serverApps.Start(startTime);
            clientApps.Start(startTime);

        } // end for b
    }

    // **************************************
    // Trace DlPhyTransmission
    // **************************************
    for (uint32_t it = 0; it != macroEnbs.GetN(); ++it) {
        Ptr < NetDevice > netDevice = macroEnbDevs.Get(it);
        Ptr < LteEnbNetDevice > enbNetDevice = netDevice -> GetObject < LteEnbNetDevice > ();
        Ptr < LteEnbPhy > enbPhy = enbNetDevice -> GetPhy();
        enbPhy -> TraceConnectWithoutContext("DlPhyTransmission", MakeBoundCallback( & MyGymEnv::GetPhyStats, myGymEnv));
    }

    // Enable output traces
    if (outputTraceFiles) {
        std::cout << "Enabling the simulation trace..." << std::endl;
        lteHelper -> EnablePhyTraces();
        lteHelper->EnableMacTraces ();
        lteHelper->EnableRlcTraces ();
        lteHelper->EnablePdcpTraces ();
    }

    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
        MakeCallback( & NotifyHandoverStartEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
        MakeCallback( & NotifyHandoverEndOkEnb));


    Simulator::Stop(Seconds(simTime));
    //        AnimationInterface anim ("animmyLTE.xml");
    //        anim.EnablePacketMetadata (true);
    //        anim.SetMaxPktsPerTraceFile(500000);
    //        anim.SetConstantPosition(remoteHost, 500,500,3);

    std::cout << "Start simulation..." << std::endl;
    Simulator::Run();

    lteHelper = 0;

    std::cout << "Waiting for simulation end..." << std::endl;
    myGymEnv -> NotifySimulationEnd();
    std::cout << "Destroying simulator..." << std::endl;
    Simulator::Destroy();

    return 0;
    std::cout << "======================================================" << std::endl;
    std::cout << "                END TOY ENVIRONMENT                   " << std::endl;
    std::cout << "======================================================" << std::endl;
}