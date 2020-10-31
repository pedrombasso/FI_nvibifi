/*
 * common_template_functions.h
 *
 *  Created on: 27/10/2019
 *      Author: fernando
 */

#ifndef COMMON_TEMPLATE_FUNCTIONS_H_
#define COMMON_TEMPLATE_FUNCTIONS_H_

#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cassert>

// #include "device_functions.h"
#include "cuda_runtime_api.h" 
#include "Parameters.h"

#ifdef OMP
#include <omp.h>
#endif

#if (__CUDACC_VER_MAJOR__ <= 7)
#include "../../mxm/half.hpp"

using half = half_float::half;
#endif

#define CHAR_CAST(x) (reinterpret_cast<char*>(x))
#define GENERATOR_MAXABSVALUE_GEMM 10
#define GENERATOR_MINABSVALUE_GEMM -GENERATOR_MAXABSVALUE_GEMM

#define GENERATOR_MAXABSVALUE_TENSOR 10
#define GENERATOR_MINABSVALUE_TENSOR -GENERATOR_MAXABSVALUE_TENSOR

#define PATH "/home/carol/pedro/fi/nvbit_release/tools/nvbitfi/test-apps/cudaTensorCoreGemm/gold.matrix"

static std::ostream& operator<<(std::ostream& os, const dim3 d) {
	os << d.x << " " << d.y << " " << d.z;
	return os;
}

template<typename T>
bool read_from_file(std::vector<T>& array) {
	std::ifstream input(PATH, std::ios::binary);
	if (input.good()) {
		input.read(CHAR_CAST(array.data()), array.size() * sizeof(T));
		input.close();
		return true;
	}
	return false;
}

__host__ bool write_to_file( std::vector<half>& array) {
  std::ofstream output(PATH, std::ios::binary);
  if (output.good()) {
    output.write(CHAR_CAST(array.data()), array.size() * sizeof(half));
    output.close();
    return true;
  }
  return false;
}

static bool exists(std::string& path) {
	std::ifstream input(path);
	auto file_exists = input.good();
	input.close();
	return file_exists;
}

// template<typename half_t, typename real_t>
// void write_abc_files(
// 		std::string& a_file_path, std::vector<half_t>& a_vector,
// 		std::string& b_file_path, std::vector<half_t>& b_vector,
// 		std::string& c_file_path, std::vector<real_t>& c_vector
// 		) {
// 	if (write_to_file(a_file_path, a_vector) == false) {
// 		throw_line(a_file_path + " could not be written\n");
// 	}

// 	if (write_to_file(b_file_path, b_vector) == false) {
// 		throw_line(b_file_path + " could not be written\n");
// 	}

// 	if (write_to_file(c_file_path, c_vector) == false) {
// 		throw_line(c_file_path + " could not be written\n");
// 	}
// }

// template<typename half_t, typename real_t>
// void read_abc_files(
// 		std::string& a_file_path, std::vector<half_t>& a_vector,
// 		std::string& b_file_path, std::vector<half_t>& b_vector,
// 		std::string& c_file_path, std::vector<real_t>& c_vector
// 		) {
// 	if (read_from_file(a_file_path, a_vector) == false) {
// 		throw_line(a_file_path + " could not be read\n");
// 	}
// 	if (read_from_file(b_file_path, b_vector) == false) {
// 		throw_line(b_file_path + " could not be read\n");
// 	}
// 	if (read_from_file(c_file_path, c_vector) == false) {
// 		throw_line(c_file_path + " could not be read\n");
// 	}
// }


template<typename real_t>
void read_gold(std::vector<real_t>& d_vector) {
	if (read_from_file(d_vector) == false) {
		//print("gold can't not be read\n");
	}
}


// static unsigned long long dmr_errors() {
// 	unsigned long long ret = 0;
// 	rad::checkFrameworkErrors(
// 			cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0,
// 					cudaMemcpyDeviceToHost));

// 	unsigned long long tmp = 0;
// 	rad::checkFrameworkErrors(
// 			cudaMemcpyToSymbol(errors, &tmp, sizeof(unsigned long long), 0,
// 					cudaMemcpyHostToDevice));

// 	return ret;
// }

template<typename real_t>
bool equals(real_t& lhs, real_t& rhs, const uint32_t threshold = 0) {
	return lhs == rhs;
}

static bool equals(half& lhs, half& rhs, const uint32_t threshold = 0) {
	return float(lhs) == float(rhs);
}

static std::ostream& operator<<(std::ostream& os, half &rhs) {
	os << float(rhs);
	return os;
}

static float fabs(half h) {
	return fabs(float(h));
}

// static bool equals(float& lhs, double& rhs, const uint32_t threshold) {
// 	assert(sizeof(float) == sizeof(uint32_t));

// 	float rhs_float = float(rhs);

// 	uint32_t lhs_data;
// 	uint32_t rhs_data;
// 	memcpy(&lhs_data, &lhs, sizeof(uint32_t));
// 	memcpy(&rhs_data, &rhs_float, sizeof(uint32_t));
// 	auto diff = SUB_ABS(lhs_data, rhs_data);

// 	return (diff <= threshold);
// }


// template<class half_t, class real_t>
std::pair<int, int> check_output_errors_dmr(std::vector<half>& gold,
		std::vector<half>& real_vector, std::vector<half>& half_vector,
		Parameters& parameter, const uint32_t threshold, const bool dmr) {
	uint32_t host_errors = 0;

#ifdef OMP
#pragma omp parallel for shared(host_errors, memory_errors)
#endif
	for (size_t i = 0; i < gold.size(); i++) {
		auto gold_value = gold[i];
		half full_precision = real_vector[i];
		half half_precision = (dmr == true) ? half_vector[i] : real_vector[i];

		//Check if DMR is OK
		
		bool dmr_equals = equals(half_precision, full_precision, threshold);


		//Is output corrupted
		bool is_output_diff = !equals(gold_value, full_precision);

		if (!equals(gold_value, full_precision)) {
#ifdef OMP
#pragma omp critical
			{
#endif

			std::stringstream error_detail("");
			error_detail << std::setprecision(20) << std::scientific;
			error_detail << "p: [" << int(floor(i / gold.size())) << ", "
					<< i % gold.size() << "], r: ";
			error_detail << full_precision;
			error_detail << ", e: " << gold_value << " smaller_precision: "
					<< half_precision;

			// if (parameter.verbose && (host_errors < 10)) {
			// 	std::cout << error_detail.str() << std::endl;

			// 	std::cout << is_output_diff << " " << !dmr_equals << std::endl;
			// }

			parameter.log_error(error_detail.str());
			host_errors++;	

#ifdef OMP
		}
#endif
		}
	}

	// auto dmr_err = dmr_errors();
	auto dmr_err = 0;

	if (dmr_err != 0) {
		std::string error_detail;
		error_detail = "detected_dmr_errors: " + std::to_string(dmr_err);
		parameter.log_info(error_detail);
	}

	parameter.update_error_count(host_errors);
	if (host_errors != 0)
		std::cout << "#";

	return {dmr_err, host_errors};
}

#endif /* COMMON_TEMPLATE_FUNCTIONS_H_ */
