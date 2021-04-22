/*****************************************************************************\

INTEL CONFIDENTIAL
Copyright 2021
Intel Corporation All Rights Reserved.

The source code contained or described herein and all documents related to the
source code ("Material") are owned by Intel Corporation or its suppliers or
licensors. Title to the Material remains with Intel Corporation or its suppliers
and licensors. The Material contains trade secrets and proprietary and
confidential information of Intel or its suppliers and licensors. The Material
is protected by worldwide copyright and trade secret laws and treaty provisions.
No part of the Material may be used, copied, reproduced, modified, published,
uploaded, posted transmitted, distributed, or disclosed in any way without
Intel's prior express written permission.

No license under any patent, copyright, trade secret or other intellectual
property right is granted to or conferred upon you by disclosure or delivery
of the Materials, either expressly, by implication, inducement, estoppel
or otherwise. Any license under such intellectual property rights must be
express and approved by Intel in writing.

File Name:  config.h

Abstract:

Notes:

\*****************************************************************************/
#pragma once

struct Config
{
    static std::string GetBlobDir()
    {
        static const std::string local_blob_dir =
            "D:/work/models/";

        static const std::string artifactory_blob_dir =
            "https://ubit-artifactory-or.intel.com/artifactory/"
            "movidius-vpu-or-local/Automation/Binaries/"
            "PreCompiled_Graph_Binaries/"
            "KMB_B0/";

        try
        {
            YAML::Node config = YAML::LoadFile( "config.yaml" );
            auto blob_dir = config[ "blob_dir" ].as< std::string >();

            return std::filesystem::exists( blob_dir ) ? blob_dir
                   : std::filesystem::exists( local_blob_dir )
                       ? local_blob_dir
                       : artifactory_blob_dir;
        }
        catch( ... )
        {
            return std::filesystem::exists( local_blob_dir )
                       ? local_blob_dir
                       : artifactory_blob_dir;
        }
    }

    template< typename T >
    static bool
    GetTestConfigValue( const char* test, const char* test_config, T& value )
    {
        try
        {
            YAML::Node config = YAML::LoadFile( "config.yaml" );
            value = config[ test ][ test_config ].as< T >();
        }
        catch( ... )
        {
            return false;
        }

        return true;
    }
};
