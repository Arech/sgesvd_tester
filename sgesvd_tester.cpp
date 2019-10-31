//#define ALSO_TEST_DENORMALS

//////////////////////////////////////////////////////////////////////////
//#include "targetver.h"
//common lib
#include <iostream>
#define STDCOUT(args) ::std::cout << args
#define STDCOUTL(args) STDCOUT(args) << ::std::endl

#include <type_traits>
#include <vector>
#include <algorithm>

//////////////////////////////////////////////////////////////////////////
//for the RNG
#include <random>
#include "AF_randomc_h/random.h"

//define a few helper classes
typedef AFog::CRandomSFMT0 baseRng_t;

template<typename real_t, typename baseRngT = baseRng_t>
struct UniRNG{
	baseRngT m_rng;

	UniRNG(int seed) : m_rng(seed) {}
	auto operator()() { return static_cast<real_t>(m_rng.Random() * 2 - 1); }
};

template<typename real_t, typename baseRngT = baseRng_t>
struct StdNormRNG{
	struct _hlpr{
		baseRngT m_rng;

		_hlpr(int seed) : m_rng(seed) {}
		//some stuff to make happy C++ named requirements: UniformRandomBitGenerator
		typedef unsigned result_type;
		static constexpr result_type min() noexcept { return ::std::numeric_limits<result_type>::min(); }
		static constexpr result_type max() noexcept { return ::std::numeric_limits<result_type>::max(); }
		result_type operator()() { return static_cast<result_type>(m_rng.BRandom()); }
	};

	_hlpr m_rng;
	::std::normal_distribution<real_t> m_distr;

	StdNormRNG(int seed) : m_rng(seed), m_distr(real_t(0.), real_t(1.)) {}
	auto operator()() { return m_distr(m_rng); }
};

//////////////////////////////////////////////////////////////////////////
#ifdef ALSO_TEST_DENORMALS
#include <xmmintrin.h>
#include <pmmintrin.h>

void inline disable_denormals()noexcept {
	unsigned int current_word = 0;
	_controlfp_s(&current_word, _DN_FLUSH, _MCW_DN);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}

void inline enable_denormals()noexcept {
	unsigned int current_word = 0;
	_controlfp_s(&current_word, _DN_SAVE, _MCW_DN);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
}

void inline report_denormals()noexcept {
	unsigned int current_word = 0;
	const auto err = _controlfp_s(&current_word, 0, 0);
	bool bDisabled = false;
	if (!err) {
		if ((current_word & _MCW_DN) == _DN_FLUSH) {
			bDisabled = true;
		}
	}

	const auto zm = _MM_GET_DENORMALS_ZERO_MODE();
	if (zm == _MM_DENORMALS_ZERO_OFF) bDisabled = false;

	const auto fm = _MM_GET_FLUSH_ZERO_MODE();
	if (fm == _MM_FLUSH_ZERO_OFF) bDisabled = false;

	STDCOUTL("Denormals are " << (bDisabled ? "disabled" : "enabled"));
}
#endif // ALSO_TEST_DENORMALS

//////////////////////////////////////////////////////////////////////////
//OpenBLAS's stuff
#include <cblas.h>
#include <complex>
#define lapack_complex_float ::std::complex<float>
#define lapack_complex_double ::std::complex<double>
#include <lapacke.h>

#pragma comment(lib,"libopenblas.dll.a")

//OpenBLAS gesvd() wrapper
template<typename fl_t>
static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, double>::value, int>
gesvd(const char jobu, const char jobvt, const int m, const int n, fl_t* A
	, const int lda, fl_t* S, fl_t* U, const int ldu, fl_t* Vt, const int ldvt, fl_t* superb)
{
	return static_cast<int>(LAPACKE_dgesvd(LAPACK_COL_MAJOR, jobu, jobvt, static_cast<lapack_int>(m), static_cast<lapack_int>(n)
		, A, static_cast<lapack_int>(lda), S, U, static_cast<lapack_int>(ldu), Vt, static_cast<lapack_int>(ldvt), superb));
}
template<typename fl_t>
static typename ::std::enable_if_t< ::std::is_same< ::std::remove_pointer_t<fl_t>, float>::value, int>
gesvd(const char jobu, const char jobvt, const int m, const int n, fl_t* A
	, const int lda, fl_t* S, fl_t* U, const int ldu, fl_t* Vt, const int ldvt, fl_t* superb)
{
	return static_cast<int>(LAPACKE_sgesvd(LAPACK_COL_MAJOR, jobu, jobvt, static_cast<lapack_int>(m), static_cast<lapack_int>(n)
		, A, static_cast<lapack_int>(lda), S, U, static_cast<lapack_int>(ldu), Vt, static_cast<lapack_int>(ldvt), superb));
}

//////////////////////////////////////////////////////////////////////////
template<typename real_t>
bool contains_NaNs(const ::std::vector<real_t>& v, bool bThrow = true) {
	for (auto e : v) {
		if (::std::fpclassify(e) == FP_SUBNORMAL) {
			if (bThrow) {
				throw ::std::exception("No NaNs expected here!");
			} else return true;
		}
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////
//some common typedefs and code
typedef ::std::make_signed_t<size_t> numel_cnt_t;

template<typename real_t>
bool is_gesvd_ok(const int rngSeed = 1490700921, const int rows = 64, const int cols = 785, const int genRetries = 3)
{
	STDCOUTL("Running is_gesvd_ok() with rngSeed=" << rngSeed << " over random [" << rows << " x " << cols << "] matrix");
	
#ifdef ALSO_TEST_DENORMALS
	report_denormals();
#endif

	const numel_cnt_t numel = static_cast<numel_cnt_t>(rows)*cols;

	//making rng
	//typedef UniRNG<real_t> Rng_t;
	typedef StdNormRNG<real_t> Rng_t;
	Rng_t rng(rngSeed);

	//allocating space for matrix [rows x cols]
	::std::vector<real_t> vMtxStor(numel);

	//need this to call gesvd
	const bool bGetU = rows >= cols;
	const auto minmn = bGetU ? cols : rows;
	::std::vector<real_t> S(2 * minmn);

	//generating matrix and calling gesvd
	int genTry;
	for (genTry = 0; genTry < genRetries; ++genTry) {
		//generating the matrix content from uniform distr in [-1,1]
		for (auto& e : vMtxStor) {
			e = rng();
		}
		contains_NaNs(vMtxStor);
		//STDCOUTL("Matrix generated, contains no NaNs");
		::std::fill(S.begin(), S.end(), real_t(0));

		//calling gesvd
		const auto r = gesvd(bGetU ? 'O' : 'N', bGetU ? 'N' : 'O', rows, cols, &vMtxStor[0], rows, &S[0]
			, static_cast<real_t*>(nullptr), rows, static_cast<real_t*>(nullptr), cols, &S[minmn]
		);

		if (r != 0) {
			STDCOUTL("gesvd() returned " << r << ", probably due to ill-conditioned data, trying to regenerate ("
				<< (genTry + 1) << "/" << genRetries <<	")");
		} else {
			STDCOUTL("gesvd() returned success, going to examine the results");
			break;
		}
	}

	if (genTry>=genRetries) {
		STDCOUTL("Failed to converge");
		return false;
	}

	if (contains_NaNs(vMtxStor, false)) {
		STDCOUTL("Returned matrix contains NaNs and that is not expected!");
		return false;
	}
	for (auto e : vMtxStor) {
		if (e > 1e20) {
			STDCOUTL("Returned matrix contains to big values and that is not expected!");
			return false;
		}
	}

	if (contains_NaNs(S, false)) {
		STDCOUTL("Returned S contains NaNs and that is not expected!");
		return false;
	}
	for (auto e : S) {
		if (e > 1e20) {
			STDCOUTL("Returned S contains to big values and that is not expected!");
			return false;
		}
	}
	
	return true;
}

void run_tests(bool& ff, bool& df) {
	STDCOUTL("Going to exec is_gesvd_ok() with real_t = float. Expecting it to fail.");
	ff = !is_gesvd_ok<float>();
	STDCOUTL((ff ? "### FAILED ###" : "passed"));

	STDCOUTL("\n\n---");

	STDCOUTL("Going to exec is_gesvd_ok() with real_t = double. It should pass.");
	df = !is_gesvd_ok<double>();
	STDCOUTL((df ? "### FAILED ###" : "passed"));
}

//////////////////////////////////////////////////////////////////////////
int main() {
#ifdef ALSO_TEST_DENORMALS
	STDCOUTL("Running tests with enabled denormals.");
#else
	STDCOUTL("Running tests.");
#endif
	STDCOUTL("Buggy OpenBLAS::gesvd() will return failure for float");

	bool ff, df;
	run_tests(ff,df);
	
#ifdef ALSO_TEST_DENORMALS
	STDCOUTL("\n\n===");
	STDCOUTL("Running tests with DISabled denormals.\nBuggy OpenBLAS::gesvd() will still return failure for float");

	disable_denormals();
	bool dff, ddf;
	run_tests(dff, ddf);
#else
	bool dff = false, ddf = false;
#endif

	STDCOUTL("\n\n===");
#ifdef ALSO_TEST_DENORMALS
	STDCOUTL("Running tests with DISabled denormals AND rounding to zero.");
#else
	STDCOUTL("Running tests with rounding to zero.");
#endif
	STDCOUTL("Buggy OpenBLAS::gesvd() will NOT return failure for float, but will produce a junk in output variables");

	unsigned int current_word = 0;
	_controlfp_s(&current_word, _RC_CHOP, _MCW_RC);

	bool zdff, zddf;
	run_tests(zdff, zddf);

	return 1 * ff + 2 * df + 4 * dff + 8 * ddf + 16 * zdff + 32 * zddf;
}

