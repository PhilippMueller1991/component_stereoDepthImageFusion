set(the_description "UbiTrack Stereo Matching")
ut_add_component(StereoMatching DEPS utcore utdataflow utvision )




ut_component_include_directories("src/StereoMatching" ${TINYXML_INCLUDE_DIR} ${LOG4CPP_INCLUDE_DIR} ${BOOSTBINDINGS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})
ut_glob_component_sources(HEADERS "src/StereoMatching/*.h" SOURCES "src/StereoMatching/*.cpp")
ut_create_single_component()
ut_install_utql_patterns()


