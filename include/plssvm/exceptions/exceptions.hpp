/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements custom exception classes derived from [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error) including source location information.
 */

#pragma once

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief Base class for all custom exception types. Forwards its message to [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)
 *        and saves the exception name and the call side source location information.
 */
class exception : public std::runtime_error {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message to [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error).
     * @param[in] msg the exception's `what()` message
     * @param[in] class_name the name of the thrown exception class
     * @param[in] loc the exception's call side information
     */
    explicit exception(const std::string &msg, std::string_view class_name = "exception", source_location loc = source_location::current());

    /**
     * @brief Returns the information of the call side where the exception was thrown.
     * @return the exception's call side information (`[[nodiscard]]`)
     */
    [[nodiscard]] const source_location &loc() const noexcept;

    /**
     * @brief Returns a sting containing the exception's `what()` message, the name of the thrown exception class and information about the call
     *        side where the exception has been thrown.
     * @return the exception's `what()` message including source location information
     */
    [[nodiscard]] std::string what_with_loc() const;

  private:
    const std::string_view class_name_;
    source_location loc_;
};

/**
 * @brief Exception type thrown if the provided file couldn't be found.
 */
class file_not_found_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit file_not_found_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if the provided file has an invalid format for the selected parser
 *        (e.g. if the arff parser tries to parse a LIBSVM file).
 */
class invalid_file_format_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit invalid_file_format_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if the requested backend is not supported on the target machine.
 */
class unsupported_backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit unsupported_backend_exception(const std::string &msg, source_location loc = source_location::current());
};

/**
 * @brief Exception type thrown if the requested kernel type is not supported.
 */
class unsupported_kernel_type_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit unsupported_kernel_type_exception(const std::string &msg, source_location loc = source_location::current());
};

}  // namespace plssvm